import numpy as np
import vicar
import ringimage as ri
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab as plb
import scipy.optimize as sciopt
import scipy.ndimage.filters as filters

# XXX IF THIS IS EVER CHANGED THEN ringimage.MakeRings needs to change, too!
NLONGITUDES_FIT = 5000                      # number of longitudes for F ring offset fitting (~0.072 deg/step)

DPR = 180./np.pi                            # Degrees per radian

def MakeFRing(radii, time, longitude_list=None):
    # Take a list of radii and possibly a list of longitudes and create
    # a list of F rings in rectangular coordinates, Saturn-centered
    fring = ri.RingImage.MakeRings(radii, time, num_longitudes=NLONGITUDES_FIT, longitude_list=longitude_list)
    
    # convert MakeRings' array from cylindrical to rectangular coordinates
    fring = ri.RingImage.Rectangular(fring, semi_maj_axes = radii)

    return fring


def MakeArray(longitude_list, radius_start, radius_end, radius_resolution):
    # Returns array of Saturn-centric (radius, longitude) in rectangular coordinates
    array = ri.RingImage.MakeRings2d(
                np.arange(radius_start, radius_end+radius_resolution, radius_resolution),
                longitude_list)
    
    return ri.RingImage.Rectangular(array)


def VicarToRingImage(vicar):
    # determine ephemeris time from vicar header
    et = vicar["IMAGE_MID_TIME"][:-1]

    # determine ISS camera from vicar header
    if vicar["INSTRUMENT_ID"] == "ISSNA":
        camera = "NAC"
    if vicar["INSTRUMENT_ID"] == "ISSWA":
        camera = "WAC"

    # get 2d array of pixel values from vicar image
    img = vicar.Get2dArray()

    # create ringimage for given ephemeris time (from imagelist) and camera
    ringim = ri.RingImage(et, camera)
    # set ringimage pixel array as the vicar image's 2d array
    ringim.SetArray(img.astype("float"))
    ringim.SetOffset((0,0))

    return ringim


def FitToFRing(ringim, fring):
    def FitFunc(offset, ringim, fring, xmax, ymax):
        """FitFunc returns the difference in brightness of the F ring compared to
        surrounding space. Offset is that which fits the created fring with the
        actual ring position in ringimage. Ringimage is created from vicar.py from
        a given ephemeris time and camera. Fring is created from vicar.py and is
        three circles generated for 3 given radii. sciopt.fmin is used on this 
        function to determine ideal offset for a given ringimage and fring.
        """
    
        # Don't let the offset get too far afield
#        print offset, xmax, ymax
        if offset[0] < -xmax*1.5:
            return -offset[0] * 1e10
        if offset[0] > xmax*1.5:
            return offset[0] * 1e10
        if offset[1] < -ymax*1.5:
            return -offset[1] * 1e10
        if offset[1] > ymax*1.5:
            return offset[1] * 1e10

        # Define the next trial offset
        ringim.SetOffset(offset)
    
        # Define the pixel array
        rays = ringim.RingToRays(fring)
        pixels = ringim.RaysToPixels(rays)
    
        # Accumulate the mean for each of three rings...
        means = [0., 0., 0.]
        for i in range(3):
    
            # Find the ring's pixels that are inside the image
            restrict = np.where(ringim.isInside(pixels[i]))
    
            # Convert the selected image coordinates to integers
            p = (pixels[i][restrict] + 0.5).astype("int")
    
            # Take the mean of all the selected pixels
            means[i] = np.mean(ringim.array[(p[:,0],p[:,1])])
    
        del rays
        rays = None
        del pixels
        pixels = None
        
        # Return off-ring mean minus the on-ring mean
        return (means[0] + means[2])/2. - means[1]

    fbest = 1.e99
    best_offset = None
    savearray = ringim.array # Same the image before we smooth it
    
    # Smooth the image.  This makes find an offset work better.
    smooth = filters.gaussian_filter(ringim.array, sigma=2, mode="constant")
    ringim.array = smooth

    pixel_size = ri.CAMERA_PIXEL_SIZE[ringim.camera]

    xmax = pixel_size * 40
    xstep = pixel_size * 20
    print xmax, xstep
    # We can get trapped in a local, but not global, minimum, so start with optimization
    # process multiple times using different initial guesses.
    for x in np.arange(-xmax, xmax+xstep, xstep):
        for y in np.arange(-xmax, xmax+xstep, xstep):
            #found_offset, residual
            result = sciopt.fmin(FitFunc, (x,y), args=(ringim, fring, xmax, xmax), disp=False,
                                                 full_output=True)
            # Save the result if it is the best so far
            if not np.isnan(result[1]) and result[1] < fbest: 
                best_offset = result[0]
                fbest = result[1]

    # set ringimage's offset
#    print 'Best residual', fbest, 'at offset', best_offset, 'radians', [best_offset[0]/ri.NAC_PIXEL_SIZE, best_offset[1]/ri.NAC_PIXEL_SIZE], 'pixels' 
    ringim.SetOffset(best_offset)
    ringim.array = savearray # Restore the original, non-smoothed image
    del smooth
    smooth = None

def NormalIF(ringim, rays):
    """change ringim from units I/F to units of Normal I/F by multiplying it by mu, the cosine of the emission angle
    """

    mu = ringim.RaysToMu(rays)

    ringim.array *= mu

def StretchIMG(ringim, xyz_array, zoom_factor, zoomarray):
    """create a stretched radius vs longitude image from a vicar image of 
    Saturn's F ring
    """

    # Map the array of ring points into the image plane
    rays   = ringim.RingToRays(xyz_array)
    pixels = ringim.RaysToPixels(rays)

    # Make an array of zeros for the stretched ring image
    stretchimg = np.zeros(xyz_array.shape[0:2])

    # Index the locations in the new array that fall inside the image
    restrict = np.where(ringim.isInside(pixels))

    # Convert the pixel coordinates to integers
    p = (zoom_factor * pixels[restrict] + 0.5).astype("int")

    # Reproject the image
    stretchimg[restrict] = zoomarray[(p[:,0],p[:,1])]

    return (stretchimg, rays)

def SliceBlack(stretchimg):
    """remove black space (pixels that do not contain image data) from 
    stretchimg
    """
    print 'check'

    # Create boolean tuple: True means nonzero values, False means all zero
    zerodetect = np.any(stretchimg, axis=0)

    # Remove black space from beginning of image
    t = np.where(zerodetect == True)
#    print t
    first_true = t[0][0]
    last_true = t[0][-1]
#    print 'check2'
    stretchimg = stretchimg[:, first_true:last_true+1]
#    print 'check3'
    return (stretchimg, first_true, last_true+1)
    
def FindOffset(vicar, ring_range):
    """
    Find the pointing offset
    """

    ringim = VicarToRingImage(vicar)
    if ringim.array.shape != (1024,1024):
        print 'ERROR! Image is not 1024x1024 - aborting' # XXX
        return None
    
    time = ringim.et
    fring = MakeFRing(ring_range, time) # Make a series of radius/longitude pairs all the way around the ring
    
    # Derive a pointing offset
    FitToFRing(ringim, fring)

    return ringim

def Reproject(vicar_data, ringim, radius_start, radius_end, radius_resolution,
              longitude_resolution, zoom_factor, zoomarray):
    
    # Now that we have the correct offset, figure out which longitudes are actually in the image
    nlong = int(360./longitude_resolution)
    all_longitude_list = 2.*np.pi/nlong * np.array(range(nlong)) #remember - longitudes are technically all Mean Anomalies (n = 1)
    fring_trial = MakeFRing(140220., ringim.et, longitude_list=all_longitude_list) # Make a sample F ring so we can restrict the longitudes to only those in the image
#    print fring_trial
    fring_cylindrical = ri.RingImage.MakeRings(140220., ringim.et, longitude_list=all_longitude_list) #need an fring in cylindrical coords (trial is already rectangular)
    rays = ringim.RingToRays(fring_trial) # Create Cassini-centered vectors to each ring position
    pixels = ringim.RaysToPixels(rays) # Convert to camera reference frame and map to pixel numbers
    restrict = np.where(ringim.isInside(pixels))
#    print restrict
#    print rays.shape, pixels.shape, all_longitude_list.shape
    valid_longitude_list = all_longitude_list[restrict[1]]
#    print valid_longitude_list
    ringim_rays = ringim.ImageToRays() # Cassini-centric rays to each image pixel
    ringim_rings = ringim.RaysToRing(ringim_rays) # Projection of rays onto ring plane returning Saturn-centric vectors

    # Calculate metadata
    phase_angle = ringim.RaysToPhase(ringim_rays)[511,511] * DPR
    incidence_angle = ringim.incidence * DPR
    emission_angle = ringim.RaysToEmission(ringim_rays)[511,511] * DPR
    
    # Convert pixels to units of normal I/F
#    NormalIF(ringim, ringim_rays)
    
    # Reproject...
#    xyz_array = MakeArray(valid_longitude_list, radius_start, radius_end, radius_resolution)
    rad_start = 140220 - radius_start
    rad_end = radius_end - 140220.
    rad_delta_vector = np.arange(-rad_start, rad_end + radius_resolution, radius_resolution)
    
    radii = fring_cylindrical[0,:,0]
    radii_restricted = radii[restrict[1]]
    true_anomalies = fring_cylindrical[0,:,1][restrict[1]]

#    xyz_array = np.empty((len(valid_longitude_list), len(rad_delta_vector), 3))
    xyz_array = np.zeros((len(rad_delta_vector), len(valid_longitude_list), 3))
    for i, long in enumerate(true_anomalies):
        xyz_array[:,i,1] = long
    xyz_array[:,:,2] = 0.0
    
    for i, rad in enumerate(radii_restricted):
        xyz_array[:,i,0] = rad + rad_delta_vector

    xyz_array = ri.RingImage.Rectangular(xyz_array) 
    (stretchimg, stretch_rays) = StretchIMG(ringim, xyz_array, zoom_factor, zoomarray)

    # Remove columns of all zero values (no image pixels)
    (stretchimg, col0, col1) = SliceBlack(stretchimg)

    # Create an array of image resolution values for the core
#    rowno = (radius_end-radius_start)/radius_resolution
    rowno = (140220. -radius_start)/radius_resolution
#    rowno = rad_avg/radius_resolutions
    print rowno
    print xyz_array.shape
    print stretch_rays.shape
    resrow = ringim.RingResolution(stretch_rays[rowno,:,:], xyz_array[rowno,:,:])
    reslist = list((resrow + 0.5).astype("int"))
    
    # Replace vicar object array with new stretchimg array
    vicar_data.SetArray(stretchimg)
#    plt.imshow(stretchimg)
#    plt.show()
    # First delete all the old versions of the keywords, if any
#    print 'b'
    for keyword in ['RADIUS', 'LONGITUDE_STEP', 'LONGITUDES_SAVED', 'PHASE_ANGLE',
                    'INCIDENCE_ANGLE', 'EMISSION_ANGLE', 'RADIAL_RESOLUTION',
                    'POINTING_OFFSET']:
        try:
            vicar_data.DeleteKeyword(keyword)
        except KeyError:
            pass
    
    # Append new keywords into vicar object header
    vicar_data.AppendKeyword("RADIUS", [radius_start, radius_end, radius_resolution])
    vicar_data.AppendKeyword("LONGITUDE_STEP", longitude_resolution)
    vicar_data.AppendKeyword("LONGITUDES_SAVED", list(valid_longitude_list*180./np.pi)) # Save in degrees
    vicar_data.AppendKeyword("PHASE_ANGLE", phase_angle)
    vicar_data.AppendKeyword("INCIDENCE_ANGLE", incidence_angle)
    vicar_data.AppendKeyword("EMISSION_ANGLE", emission_angle)
    vicar_data.AppendKeyword("RADIAL_RESOLUTION", reslist)
    vicar_data.AppendKeyword("POINTING_OFFSET", list(ringim.offset))
#    print 'c'
    return stretchimg
