import ringutil
import cspice
import ckloader
import spkloader
import numpy as np
import math
import os
import scipy.optimize as sciopt

################################################################################
# Load the most import SPICE kernels.
################################################################################

SPICE_PATH = os.getenv('SPICE_PATH')

# Leapseconds
cspice.furnsh(os.path.join(SPICE_PATH, "General/LSK/naif0010.tls"))

# Planetary constants and frames
cspice.furnsh(os.path.join(SPICE_PATH, "General/PCK/pck00010.tpc"))
cspice.furnsh(os.path.join(SPICE_PATH, "Cassini/PCK/cpck19Feb2014.tpc"))
cspice.furnsh(os.path.join(SPICE_PATH, "Cassini/FK/cas_rocks_v18.tf"))

# Cassini spacecraft kernels
cspice.furnsh(os.path.join(SPICE_PATH, "Cassini/SCLK/cas00158.tsc"))
cspice.furnsh(os.path.join(SPICE_PATH, "Cassini/IK/cas_iss_v10.ti"))
cspice.furnsh(os.path.join(SPICE_PATH, "Cassini/FK/cas_v40.tf"))

# Ephemeris kernels
cspice.furnsh(os.path.join(SPICE_PATH, "General/SPK/de430.bsp"))
cspice.furnsh(os.path.join(SPICE_PATH, "Saturn/SPK/sat357-rocks.bsp"))
cspice.furnsh(os.path.join(SPICE_PATH, "Saturn/SPK/sat357.bsp"))
#cspice.furnsh(SPICE_PATH + "Saturn/SPK/sat317.bsp")
cspice.furnsh(os.path.join(SPICE_PATH, "Saturn/SPK/sat363.bsp"))

# Cassini paths to C and SP kernels
CK_PATH  = os.path.join(SPICE_PATH, "Cassini/CK-reconstructed/")
SPK_PATH = os.path.join(SPICE_PATH, "Cassini/SPK-reconstructed/")

################################################################################
# SPICE IDs
################################################################################

SSB_ID        = cspice.bodn2c("SSB")
SUN_ID        = cspice.bodn2c("SUN")
SATURN_ID     = cspice.bodn2c("SATURN")
CASSINI_ID    = cspice.bodn2c("CASSINI")
PANDORA_ID    = cspice.bodn2c("PANDORA")
PROMETHEUS_ID = cspice.bodn2c("PROMETHEUS")

NAC_FRAME_NAME = "CASSINI_ISS_NAC"
WAC_FRAME_NAME = "CASSINI_ISS_WAC"

NAC_FRAME_ID   = cspice.bodn2c(NAC_FRAME_NAME)
WAC_FRAME_ID   = cspice.bodn2c(WAC_FRAME_NAME)

# Should be defined in bodn2c, but isn't (till now)
CASSINI_FRAME_ID = CASSINI_ID * 1000
cspice.boddef("CASSINI_SC_FRAME", CASSINI_FRAME_ID)

################################################################################
# Kernel Loaders
################################################################################

CASSINI_CKLOADER  = ckloader.CKloader(CK_PATH, CASSINI_FRAME_ID)
CASSINI_SPKLOADER = spkloader.SPKloader(SPK_PATH, CASSINI_ID)

################################################################################
# Create the rotation matrix from J2000 to the inertial Saturn frame. This is
# fixed for all calculations.
################################################################################

# Reference time
REFERENCE_DATE = "1 JANUARY 2007"       # This is the date Doug used. It is only
                                        # used to define the instantaneous pole.
REFERENCE_ET = cspice.utc2et(REFERENCE_DATE)

# Coordinate frame:
#   Z-axis is Saturn's pole;
#   X-axis is the ring plane ascending node on J2000
j2000_to_iau_saturn = cspice.pxform("J2000", "IAU_SATURN", REFERENCE_ET)

saturn_z_axis_in_j2000 = cspice.mtxv(j2000_to_iau_saturn, (0,0,1))
saturn_x_axis_in_j2000 = cspice.ucrss((0,0,1), saturn_z_axis_in_j2000)

J2000_TO_SATURN = cspice.twovec(saturn_z_axis_in_j2000, 3,
                                saturn_x_axis_in_j2000, 1)

SATURN_TO_J2000 = J2000_TO_SATURN.transpose()

################################################################################
# Define the NAC and WAC cameras each as an array of vectors in the instrument
# frame.
################################################################################
# From Cassini/IK/cas_iss_v10.ti...
#
#                                    ^
#              Samples               | Ycm
#    (1,1) + + + >                   |
#      +  \_____________________________________________________  ___
#      +  |                                                     |  |
#      +  |                Samples                              |  |
#      V  |      (1,1) + + + >                                  |  |
# Lines   |        +  \_____________________________  ___       |  |
#         |        +  |                             |  |        |  |
#         |        +  |                             |  |        |  |
#         |        V  |                             |  |        |  |
#         |   Lines   |                             |  |        |  |
#         |           |                             |  |        |  |
#    <--- |           |              x              | 0.35      | 3.48
#  Xcm    |           |                Zcm          |  |        |  |
#         |           |                             |  |        |  |
#         |           |                             |  |        |  |
#         |           |                             |  |        |  |
#         |           |_____________________________| _|_       |  |
#         |            NAC                                      |  |
#         |           |------------0.35-------------|           |  |
#         |                                                     |  |
#         |_____________________________________________________| _|_
#          WAC
#         |------------------------3.48-------------------------|
#
################################################################################

NPIXELS = 1024
NAC_FOV_DEGREES = 0.35
WAC_FOV_DEGREES = 3.48

NAC_PIXEL_SIZE = 0.35 / NPIXELS * np.pi/180.
WAC_PIXEL_SIZE = 3.48 / NPIXELS * np.pi/180.

# Useful dictionaries
CAMERA_FRAME_NAME = {"NAC":NAC_FRAME_NAME, "WAC":WAC_FRAME_NAME}
CAMERA_FRAME_ID   = {"NAC":NAC_FRAME_ID,   "WAC":WAC_FRAME_ID}
CAMERA_PIXEL_SIZE = {"NAC":NAC_PIXEL_SIZE, "WAC":WAC_PIXEL_SIZE}

eccen = 0.00235
w0 = 24.2                        #deg
dw = 2.70025                

def saturn_to_pandora(et):
    '''
    Return Saturn->Pandora vector
    '''
    CASSINI_CKLOADER.furnishET(et)
    try:
        (pandora_j2000, lt) = cspice.spkez(PANDORA_ID, et, "J2000", "LT+S", SATURN_ID)
        dist = np.sqrt(pandora_j2000[0]**2.+pandora_j2000[1]**2.+pandora_j2000[2]**2.)
        longitude = math.atan2(pandora_j2000[0], pandora_j2000[1]) * 180./np.pi
        if longitude < 0:
            longitude += 360
        longitude = ringutil.InertialToCorotating(longitude, et)
        return (dist, longitude)
    except:
        return None

def saturn_to_prometheus(et):
    '''
    Return Saturn->Prometheus vector
    '''
    CASSINI_CKLOADER.furnishET(et)
#    try:
    if True:
        (prometheus_j2000, lt) = cspice.spkez(PROMETHEUS_ID, et, "J2000", "LT+S", SATURN_ID)
        prometheus_sat = np.dot(J2000_TO_SATURN, prometheus_j2000[0:3])
        dist = np.sqrt(prometheus_sat[0]**2.+prometheus_sat[1]**2.+prometheus_sat[2]**2.)
        longitude = math.atan2(prometheus_sat[1], prometheus_sat[0]) * 180./np.pi
        if longitude < 0:
            longitude += 360
#        print prometheus_sat, dist, longitude
        longitude = ringutil.InertialToCorotating(longitude, et)
        return (dist, longitude)
#    except:
#        return None

#HELPER FUNCTIONS
def keplers_equation_resid(p, M, e):
# p = (E)
# params = (M, e)
    return np.sqrt((M - (p[0]-e*np.sin(p[0])))**2)

# Mean anomaly M = n(t-tau)
# Find E, the eccentric anomaly, the angle from the center of the ellipse and the pericenter to a
# circumscribed circle at the place where the orbit is projected vertically.
def find_E(M, e):
    result = sciopt.fmin(keplers_equation_resid, (0.,), args=(M, e), disp=False, xtol=1e-20, ftol=1e-20)
    return result[0]

# Find r, the radial distance from the focus, and f, the true anomaly, the angle from the focus and the
# pericenter to the orbit.
def rf_from_E(a, e, E):
    f_list = []
#    E_angles = (E + 2*np.pi)%(2*np.pi)
#    f_list = np.arctan(2*(np.sqrt(1+e)/np.sqrt(1-e))*np.tan(E_angles/2.))
    for i,angle in enumerate(E):
        f = np.arccos((np.cos(angle)-e)/(1-e*np.cos(angle)))
        if angle > np.pi:
            f = 2*np.pi - f
        f_list.append(f)               #moves the angle back to the correct coordinate
    f_list = np.array(f_list)
    
    return np.array([a*(1-e*np.cos(E)), f_list])   
#    return np.array([a*(1-e*np.cos(E)), np.arccos((np.cos(E)-e)/(1-e*np.cos(E)))])

# Find X,Y (relative to the focus)
def xy_from_E(a, e, E):
    return np.array([a*(np.cos(E)-e), a*np.sqrt(1-e**2)*np.sin(E)])

# Find X,Y (relative to the origin)
def xy_from_E_origin(a, e, E):
    return np.array([a*np.cos(E), a*np.sqrt(1-e**2)*np.sin(E)])

################################################################################
# Object Class...
################################################################################

class RingImage():
    """This object encapsulates the key information about a typical image.
    Optionally, it can also hold the image pixel array."""

    def __init__(self, midtime, camera, array=None, offset=(0.,0.)):
        """Constructor.

        Input:
            midtime         The mid-time of the image, as a character string or
                            a number. A string is assumed to be UTC; a number is
                            interpreted as ET.

            camera          The name of the camera, either "NAC" or "WAC".

            array           Optional array of pixels, dimensioned [line,sample].
                            Assumed to be 1024 x 1024. This can be filled in
                            later or not at all if preferred.

            offset          Optional offset in camera frame (x,y) coordinates
                            to be added to all pixel direction vectors. Default
                            (0., 0.). This can be filled in later if preferred.
        """

        # Interpret the time
        if type(midtime) == type(""):
            self.et = cspice.utc2et(midtime)
        else:
            self.et = midtime

        # Load the kernels if necessary
        CASSINI_CKLOADER.furnishET(self.et)
        CASSINI_SPKLOADER.furnishET(self.et)

        # Save key parameters about the camera
        self.camera = camera

        self.frame_name = CAMERA_FRAME_NAME[camera]
        self.frame_id   = CAMERA_FRAME_ID[camera]
        self.pixel_size = CAMERA_PIXEL_SIZE[camera]
        self.npixels    = np.array((NPIXELS, NPIXELS))  # lines, samples
        self.midpixel   = (self.npixels - 1.)/2.        # lines, samples

        # Save the array, if any
        self.array = array

        # Locate the apparent position of Saturn relative to Cassini
        # Allow for light travel time and stellar aberration
        (saturn_j2000, lt) = cspice.spkez(SATURN_ID, self.et, "J2000", "LT+S",
                                          CASSINI_ID)
        self.saturn_et = self.et - lt

        # Transform Saturn's position to the _saturn inertial frame
        saturn_sat = np.dot(J2000_TO_SATURN, saturn_j2000[0:3])
    
        # Save the reversed position to put Saturn at the origin
        self.cassini_sat = -saturn_sat
    
        # Generate the rotation matrix from the camera frame to the Saturn frame
        camera_to_j2000 = cspice.pxform(CAMERA_FRAME_NAME[camera], "J2000",
                                        self.et)
        self.camera_to_saturn = np.dot(J2000_TO_SATURN, camera_to_j2000)

        # The inverse is also handy
        self.saturn_to_camera = self.camera_to_saturn.transpose()
        
        # Assume the pointing offset is zero. This can be overridden later
        self.offset = np.array(offset)                  # (x,y) in camera frame
        self.imagerays = None

        # Locate the apparent position of the Sun relative to Saturn
        # Allow for light travel time and stellar aberration
        (sun_j2000, lt) = cspice.spkez(SUN_ID, self.et, "J2000", "LT+S",
                                       SATURN_ID)

        # Transform the Sun's position to a unit vector in the _saturn inertial
        norm = np.sqrt(np.dot(sun_j2000, sun_j2000))
        self.sun_sat = np.dot(J2000_TO_SATURN, sun_j2000[0:3]) / norm

        # Determine the incidence angle on the ring plane
        self.incidence = np.arccos(np.abs(self.sun_sat[2]))
        self.sun_sign = np.sign(self.sun_sat[2])

    def SetOffset(self, offset):
        """Sets the pointing offset in camera frame (x,y) coordinates."""

        if offset == None:
            self.offset = None
        else:
            self.offset = np.array(offset)
        self.imagerays = None

    def SetArray(self, array):
        """Sets the image pixel array, which must be 1024 x 1024 (for now)."""

        self.array = array

    def isInside(self, pixels, slop=0.):
        """Returns a boolean array indicating True if the pixel coordinates fall
        inside the image, and false if they do not. The "slop" parameter defines
        a perimeter distance in units of pixels where coordinates should be
        treated as if they are inside. This can be used to allow for possible
        uncertainty in the precise pointing of an image, so that we do not
        exclude features that "might" be in the image."""

        # Reshape the array for convenience
        pixels2 = pixels.reshape((pixels.size/2,2))

        # Create result array
        result = np.where((pixels2[:,0] < -slop - 0.5)           |
                          (pixels2[:,1] < -slop - 0.5)           |
                          (pixels2[:,0] >= self.npixels[0] + slop - 0.5) |
                          (pixels2[:,1] >= self.npixels[1] + slop - 0.5),
                          False, True)

        # Reshape result and return
        newshape = list(pixels.shape)
        newshape = tuple(newshape[0:-1])
        return result.reshape(newshape)

    def RaysToPixels(self, rays):
        """Returns an array of (line,sample) pixel coordinates in the camera,
        based on a set of rays provided in _saturn inertial coordinates."""

        # Reshape the vector array for convenience. New shape is (N,3).
        rays_sat = rays.reshape((rays.size/3,3))

        # Convert from _saturn inertial coordinates to _camera coordinates
        rays_cam = np.dot(self.saturn_to_camera, rays_sat.transpose())
        # Note: Shape is now (3,N)

        # Features located "behind" the camera are moved to the edge of the
        # image plane. Otherwise, they might inadvertently end up back in the
        # field of view.
        rays_cam[2,:] = np.maximum(rays_cam[2,:], 0.)

        # Convert rays to unit length
        rays_cam /= np.sqrt(rays_cam[0,:]**2 + rays_cam[1,:]**2
                                             + rays_cam[2,:]**2)

        # Transpose, slice and apply offset
        rays_cam = rays_cam.transpose()[:,0:2] - self.offset
        # Note: Shape is now (N,2)

        # Convert to pixel coordinates (line,sample)
        # Note that the last coordinates must be swapped to get the correct
        # order
        pixels = self.midpixel - rays_cam[:,1::-1]/self.pixel_size

        # Get the new shape (with last dimension 2 instead of 3)
        newshape = list(rays.shape)
        newshape[-1] = 2
        newshape = tuple(newshape)

        # Reshape and return
        return pixels.reshape(newshape)

    def PixelsToRays(self, pixels):
        """Returns an array of pointing vectors in saturn inertial coordinates,
        one for each pixel provided in an array of (line,sample) coordinates.
        The vectors are returned with unit length."""

        # Reshape the vector array for convenience. New shape is (N,2).
        pixels2 = pixels.reshape((pixels.size/2,2))

        # Create an empty array of vectors
        rays_cam = np.empty((pixels2.shape[0],3), dtype="float")

        # Fill in the vectors using camera coordinates
        rays_cam[:,0] = self.pixel_size * pixels2[1] + self.offset[0]

        rays_cam[:,1] = np.transpose(
                        self.pixel_size * pixels2[0] + self.offset[1])

        rays_cam[:,2] = np.sqrt(1. - rays_cam[:,0]**2 - rays_cam[:,1]**2)

        # Convert from _camera coordinates to _saturn inertial coordinates
        rays_sat = np.dot(self.camera_to_saturn, rays_cam.transpose())
        # Note: Shape is now (3,N)

        # Reshape and return
        newshape = list(pixels.shape)
        newshape[-1] = 3
        newshape = tuple(newshape)

        return rays_sat.reshape(newshape)

    def ImageToRays(self):
        """Returns an array of pointing vectors, one for each image pixel.
        This array is cached in case it is needed again. The array is
        dimensioned (1024,1024,3), where vectors are given in saturn inertial
        coordinates."""

        # If the array is already cached, return it
        if self.imagerays != None: return self.imagerays

        # Generate the rays in the camera frame using the  offset values
        x_offsets = np.arange(self.midpixel[1], -self.midpixel[1]-1., -1.)
        y_offsets = np.arange(self.midpixel[0], -self.midpixel[0]-1., -1.)

        rays_cam = np.empty((self.npixels[0], self.npixels[1], 3))

        rays_cam[:,:,0] = self.pixel_size * x_offsets + self.offset[0]
        # Note shapes: (1024,1024) = constant * (1024) + constant

        rays_cam[:,:,2] = self.pixel_size * y_offsets + self.offset[1]
        rays_cam[:,:,1] = rays_cam[:,:,2].transpose()
        # Note that we use layer 2 as a temporary buffer, because attempting
        # to transpose layer 1 atop itself creates errors.

        rays_cam[:,:,2] = np.sqrt(1. - rays_cam[:,:,0]**2 - rays_cam[:,:,1]**2)

        # Rotate the vectors to the Saturn inertial frame.
        # The dot operator works on the second-to-last index of the second
        # argument, so we need to swap some axes first

        rolled = np.rollaxis(rays_cam, 2, 1)
        # New shape: (1024,3,1024)

        self.imagerays = np.dot(self.camera_to_saturn, rolled)
        # New shape: (3,1024,1024)

        # Restore to proper axis order
        self.imagerays = np.rollaxis(self.imagerays, 0, 3)

        return self.imagerays

    def RaysToRing(self, rays):
        """This function projects a set of rays into the ring plane. It returns
        an array of Saturn-centered vectors. In general, the z-coordinate will
        be zero or at least very small. Coordinate values of NaN are inserted for
        cases where the pixel's line of sight does not intersect the ring
        plane."""

        # Reshape the rays for convenience
        rays2 = rays.reshape((rays.size/3,3))

        # Find the point R along each line of sight where z = 0.
        #   R = Cassini - (Cassini_z/LOS_z) LOS

        rays2 = rays2.transpose()
        # Note shape: (3,N)

        # Determine the distances to the ring intercept point
        distance = -self.cassini_sat[2] / rays2[2,:]

        # Fill in not-a-number
        distance = np.where(distance >= 0., distance, np.nan)

        scaled = distance * rays2
        # Note shapes: (N) * (3,N) yields (3,N)

        result = self.cassini_sat + scaled.transpose()
        # Note shapes: (3) + (N,3) yields (N,3)

        # Return the result, restored to its original shape
        return result.reshape(rays.shape)

    def RingResolution(self, rays, rings):
        """This function projects a set of rays into the ring plane. It returns
        the local ring resolution in units of km/pixel at each pixel. Values of
        NaN are inserted where the ray does not intersect the ring plane.

        Input:
            rays        the same collection of rays passed to RaysToRing().
            rings       the ring radius vectors returned by RaysToRing().

        Return:         An array of the same shape as radii, containing the
                        value of radial resolution in km/radian.
        """

        # Find the point R along each line of sight where z = 0.
        #   R = Cassini - (Cassini_z/LOS_z) LOS
        # where the vectors are expressed in Saturn-centered coordinates with
        # the z-axis parallel to Saturn's pole.
        #
        # First solve for the gradient vector of R^2.
        #   Define Cassini = (Cx,Cy,Cz) and LOS = (x,y,z)
        #
        #   R^2 = C^2 - 2 (Cz/z) dot(C,LOS) + (Cz/z)^2 LOS^2
        #
        #   R^2 = C^2 - 2 (Cz/z) (x Cx + y Cy + z Cz)
        #             + (Cz/z)^2 (x^2 + y^2 + z^2)
        #
        #   R^2 = C^2 - 2 Cz (x Cx + y Cy)/z - 2 Cz^2
        #             + Cz^2 (x^2 + y^2)/z^2 + Cz^2
        #
        #   dR^2/dx = -2 Cz Cx/z + 2 Cz^2 x/z^2
        #           = 2 Cz/z^2 (x Cz - z Cx)
        #
        #   dR^2/dy = -2 Cz Cx/z + 2 Cz^2 y/z^2
        #           = 2 Cz/z^2 (y Cz - z Cy)
        #
        #   dR^2/dz = 2 Cz/z^2 [(x Cx + y Cy) - Cz (x^2 + y^2)/z]
        #
        # Note that this gradient vector satisfies the property that its dot
        # product with (x,y,z) is zero. This is necessary because the intercept
        # point on the ring plane does not change if LOS is multiplied by an
        # arbitrary constant. Only changes in LOS that are perpendicular to LOS
        # will move the ring intercept point.
        #
        # Convert to a gradient in R.
        #   dR^2/ds = 2R dR/ds  (for s = x, y or z)
        #
        #   dR/ds = dR^2/ds / (2R)
        #
        # The radial resolution is the magnitude of the gradient.
        #
        #   resolution = sqrt[(dR/dx)^2 + (dR/dy)^2 + (dR/dz)^2]
        #              = sqrt[(dR^2/dx)^2 + (dR^2/dy)^2 + (dR^2/dz)^2] / 2R
        #
        #              = Cz/(R z^2) sqrt[ (x Cz - z Cx)^2 +
        #                                 (y Cz - z Cy)^2 +
        #                                ((x Cx + y Cy) - Cz (x^2 + y^2))^2]
#        print rays.shape
#        
#        print rings.shape
#        print rings
        # Reshape the rays for convenience
        print 'a'
        rays2 = rays.reshape((rays.size/3,3)).transpose()
        # Note shape: (3,N)

        # Derive the radius values
       
        rings2 = rings.reshape((rings.size/3,3))
        radii = np.sqrt(np.sum(rings2**2, axis=-1))
        # Note shape: (N)
        print 'b'
        # Create gradient output array
        grad = np.empty(rays2.shape)

        grad[0] = (rays2[0] * self.cassini_sat[2] -
                   rays2[2] * self.cassini_sat[0])

        grad[1] = (rays2[1] * self.cassini_sat[2] -
                   rays2[2] * self.cassini_sat[1])

        grad[2] = (rays2[0] * self.cassini_sat[0] +
                   rays2[1] * self.cassini_sat[1]
                 - self.cassini_sat[2] * (rays2[0]**2 + rays2[1]**2)/rays2[2])
        print 'c'
        # Determine its magnitude and rescale
        res = np.sqrt(np.sum(grad**2, axis=0))
        res *= self.pixel_size * np.abs(self.cassini_sat[2]) / (radii * rays2[2]**2)
        print 'd'
        return res.reshape(rays.shape[0:-1])

    def RingToRays(self, xyz_sat):
        """This returns a set of Cassini-centered unit rays in Saturn inertial
        coordinates, based on a set of Saturn-centered locations. The locations
        are typically in the ring plane (z=0) but this is not required."""

        # Calculate the offset vectors from Cassini to the points
        rays_sat = xyz_sat - self.cassini_sat

        # Reshape vector array for convenience. New shape is (N,3).
        rays_sat = rays_sat.reshape((rays_sat.size/3,3))

        # Convert rays to unit length. This requires a transpose.
        rays_sat = rays_sat.transpose()
        rays_sat /= np.sqrt(rays_sat[0,:]**2 + rays_sat[1,:]**2
                                             + rays_sat[2,:]**2)
        # Note shape: (3,N) /= (N) yields (3,N)

        # Transpose, restore to original shape and return
        rays_sat = rays_sat.transpose()
        return rays_sat.reshape(xyz_sat.shape)

    def RaysToPhase(self, rays):
        """This function returns an array of phase angle values for an array of
        unit camera rays."""

        return np.arccos(-np.dot(rays,self.sun_sat))

    def RaysToEmission(self, rays):
        """This function returns an array of emission angle values for an array
        of unit camera rays."""

        return np.arccos(-self.sun_sign * rays[...,2])

    def RaysToMu(self, rays):
        """This function returns the mu-factor for a set of camera rays."""

        return np.abs(rays[...,2])

    @staticmethod
    def Cylindrical(array):
        """Converts an array of rectangular coordinates (x,y,z) into an
        identically shaped array in which the values are in cylindrical
        coordinates.
    
        Input:
            array       an arbitrary array in which the last axis has size 3.
    
        Return:         an identically shaped array in which the last axis has
                        been replaced by values of (radius, longitude, z).
                        Longitude ranges from zero to 2*pi.
        """
    
        # Make a version of the array of shape N x 3
        source = array.reshape((array.size/3,3))
    
        # Make an empty result array with the same shape
        result = np.empty(source.shape, dtype="float")
    
        # Fill in the radii
        result[:,0] = np.sqrt(source[:,0]**2 + source[:,1]**2)
    
        # Fill in the longitudes
        result[:,1] = np.arctan2(source[:,1], source[:,0]) % (2.*np.pi)
    
        # Fill in Z
        result[:,2] = source[:,2]
    
        # Reshape result
        return result.reshape(array.shape)

    @staticmethod
    def Rectangular(array, semi_maj_axes = None):
        """Converts an array of cylindrical coordinates (radius,longitude,z)
        into an identically shaped array in which the values are in rectangular
        coordinates.

        Input:
            array       an arbitrary array in which the last axis has size 3.
    
        Return:         an identically shaped array in which the last axis has
                        been replaced by values of (x,y,z).
                        
        EDIT: Now uses elliptical transformations.
        x = a*cos(theta)
        y = b*sin(theta)
        theta = Eccentric angle = atan((a/b)*tan(longitude)) (see wikipedia's explanation of parametric ellipse)
        b = sqrt(a**2(1-e**2))
        """
        # Make a version of the array of shape N x 3
#        eccen = 0.00235
        source = array.reshape((array.size/3,3))
#        print source.shape
        # Make an empty result array with the same shape
        result = np.empty(source.shape, dtype="float")
        
#        print source.shape
#        if semi_maj_axes != None:
#            semi_maj_axes = np.array(semi_maj_axes)
#            #need to get the semi major axes to have the same length as array
##            print source.shape[0], semi_maj_axes.size
#            semi_maj_axes = np.array(sorted(np.tile(semi_maj_axes, source.shape[0]/semi_maj_axes.size)))
#            
#            semi_min_axes = np.sqrt((semi_maj_axes**2)*(1-eccen**2))
##            print semi_min_axes.shape, semi_maj_axes.shape
#            axes_ratios = semi_maj_axes/semi_min_axes
#            
#            result = np.empty(source.shape, dtype="float")
#            
#            #We need to make sure that arctan is mapping to the correct coordinate system.
#            #We know that t and longitude will always be in the same quadrant 
#            phi = source[:,1]
#            true_longs = (phi + 2*np.pi)%(2*np.pi)
#        
#            arctan_angle = np.arctan(axes_ratios*np.tan(true_longs))
#            for i,angle in enumerate(phi):
#                if (np.pi/2.) < angle <= (3.*np.pi/2.):
#                    arctan_angle[i] += np.pi               #moves the angle back to the correct coordinate
#            
#            #Fill in X and Y
#            result[:,0] = semi_maj_axes * np.cos(arctan_angle)
#            result[:,1] = semi_min_axes * np.sin(arctan_angle)
#            result[:,2] = source[:,2]
#            
#        else:  
#            #the reprojection part acts differently
#            #using the 'radii" as semi-major axes means there should be one semi-minor axis for each "radius"
#            #radii are source[:,0]
#            
#            semi_maj_axes = source[:,0]
#            semi_min_axes = np.sqrt((semi_maj_axes**2)*(1-eccen**2))
#            axes_ratios = semi_maj_axes/semi_min_axes
#            
#            result = np.empty(source.shape, dtype = 'float')
#            
#            phi = source[:,1]
#            true_longs = (phi + 2*np.pi)%(2*np.pi)
#            arctan_angle = np.arctan(axes_ratios*np.tan(true_longs))
#            for i,angle in enumerate(phi):
#                if (np.pi/2.) < angle <= (3.*np.pi/2.):
#                    arctan_angle[i] += np.pi               #moves the angle back to the correct coordinate
#                    
#            #Fill in X and Y
#            result[:,0] = semi_maj_axes * np.cos(arctan_angle)
#            result[:,1] = semi_min_axes * np.sin(arctan_angle)
#            result[:,2] = source[:,2]
#            

#        print 'A'
#         Fill in X
        result[:,0] = source[:,0] * np.cos(source[:,1])
        
#         Fill in Y
        result[:,1] = source[:,0] * np.sin(source[:,1])
        
        # Fill in Z
        result[:,2] = source[:,2]
#        print 'B'
    # Reshape result
        return result.reshape(array.shape)

    @staticmethod
    def MakeRings(a_arr, time, num_longitudes=None, longitude_list=None, z=0.):
        """Returns an array of longitude points in Saturn-centered cylindrical
        coordinates, based on an arbitrary set of radii and a specified number
        of steps in longitude.  Note the steps in 'longitude' are really steps in
        Mean Longitude!!"""
        # Make sure the radius list is an array
        a_arr = np.array(a_arr)
        
        #radii is a tuple of three semi-major axes (inner, core, outer)
        #elliptical radii array constants 
        time = time/86400. #changes the ET time in seconds to days
        w = (w0 + dw*time)*(np.pi/180.)    #radians (longitude array is in radians)
        print 'LONGITUDE OF PERICENTER', w
        # Create an array of longitudes (longitudes now in terms of Mean Longitude (M))
        #now we need to move Eccentric Anomaly to correspond to Mean Longitude in Saturn's Reference frame.
        # Lambda = M - w
        #Therefore, E in terms of lambda is E_lambda = E rotated by -w/(360./num_longitdues)
        if longitude_list == None:
            longitude_list = 2.*np.pi/num_longitudes * np.array(range(num_longitudes))
            assert num_longitudes == 5000
            E_path = os.path.join(ringutil.ROOT, 'eccentric_anom_arr_5000.npy') # XXX THIS IS REALLY DANGEROUS MAKE SURE IT's ALWAYS 5000!
            w = w%(2*np.pi)
            rotate = int(w/(2*np.pi/num_longitudes))
            E_list = np.load(E_path)
            E_list = np.roll(E_list, rotate)
            print E_list
        else:
            assert len(longitude_list) == 18000
            w = w%(2*np.pi)
            rotate = int(w/(2*np.pi/len(longitude_list)))
            print rotate
            E_path = os.path.join(ringutil.ROOT, 'eccentric_anom_arr_18000.npy')
            E_list = np.load(E_path)
            E_list = np.roll(E_list, rotate)
    
            print E_list
        #find the Eccentric Anomaly
#        E_list = []
#        for m in longitude_list:
#            E = find_E(m, eccen)
#            E_list.append(E)
#            
#        E_list = np.array(E_list)
#        print max(E_list), min(E_list)
        #happens if radii is a float or an integer (cannot index a 0-d array)
        if a_arr.shape == ():
#            radii_set_list = radii*(1-eccen**2)/(1 + eccen*np.cos(long_arr - w))
            radii_set_list, f_list = rf_from_E(a_arr, eccen, E_list) 
            f_list += w
            f_list %=(2*np.pi)
        else:
            radii_set_list = list(np.zeros((len(a_arr))))
            
            for i, a in enumerate(a_arr):
                r_set, f_list = rf_from_E(a, eccen, E_list) #returns a set of radii for one semi-major axis
                radii_set_list[i] = r_set
                f_list += w
                f_list %=(2*np.pi)
#                radii_set_list[i] = radii_set*(1-eccen**2)/(1 + eccen*np.cos(long_arr - w))
            
        radii_set_list = np.array(radii_set_list)
#        radii = np.array([radii_inner, radii_core, radii_outer])
        
        # Create an empty array of the proper size
        if a_arr.shape ==():
            result = np.empty((a_arr.size, longitude_list.size, 3))
        else:
            result = np.empty((len(a_arr), longitude_list.size, 3))
        
#        print max(f_list), min(f_list)
#        print f_list
        # Fill in longitudes and elevations
        result[:,:,1] = f_list
        result[:,:,2] = z
        if a_arr.shape == ():
#            print 'using ()'
            result[0,:,0] = radii_set_list
        else:
#            print 'using normal case'
            for i, radii_set in enumerate(radii_set_list):
                result[i,:,0] = radii_set
                
#        result[1,:,0] = radii[1]
#        result[2,:,0] = radii[2]

        # Roll axes to fill in radii
#        result = np.rollaxis(result, 1, 0)
#        result[:,:,0] = radii.reshape(np.size(radii))
#        result = np.rollaxis(result, 1, 0)
#
#        # Re-shape to allow for dimensionality of radii
#        if radii.size == 1:
#            newshape = (longitude_list.size,3)
#        else:
#            newshape = list(radii.shape) + [longitude_list.size, 3]
#            newshape = tuple(newshape)
#
#        return result.reshape(newshape)

        return result

    @staticmethod
    def MakeRings2d(radrange, lonrange, z=0.):
        """Returns a 2-D array of (radius,longitude) points in Saturn-centered
        cylindrical coordinates, where the first axis increases with radius, the second axis increases with longitude."""

        # Create an empty array of the proper size
        result = np.empty((np.size(radrange), np.size(lonrange), 3))

        # Fill in longitudes and elevations
        result[:,:,1] = np.asfarray(lonrange)
        result[:,:,2] = z

        # Roll axes to fill in radii
        result = np.rollaxis(result, 1, 0)
        result[:,:,0] = np.asfarray(radrange)
        result = np.rollaxis(result, 1, 0)

        return result

################################################################################
# Test program for COISS_2053/data/1613172417_1613290696/W1613172417_1.IMG
#
# IMAGE_MID_TIME = 2009-043T22:46:38.791
################################################################################

def test():

    ringim = RingImage("2009-043T22:46:38.791", "WAC")#, offset=(1.e-3, -2.e-3))

    print "ImageToRays..."
    rays = ringim.ImageToRays()

    print ""
    print "X, Y, Z..."
    print rays

    print ""
    print "RaysToRing..."
    vectors = ringim.RaysToRing(rays)
    cyl = RingImage.Cylindrical(vectors)
    cyl[:,:,1] *= 180./np.pi

    print ""
    print "Radius, longitude, elevation..."
    print cyl

    cyl[:,:,1] /= 180./np.pi
    vectors = RingImage.Rectangular(cyl)

    print ""
    print "RingToRays..."
    rays = ringim.RingToRays(vectors)
    print rays

    print ""
    print "RaysToPixels..."
    pixels = ringim.RaysToPixels(rays)

    print ""
    print "Line, sample..."
    print pixels

    print ""
    print "MakeRings..."
    fring = RingImage.MakeRings(140220., 10000)

    cyl = fring.copy()
    cyl[:,1] *= 180./np.pi

    print ""
    print "Radius, longitude, elevation..."
    print cyl

    fring = RingImage.Rectangular(fring)

    rays = ringim.RingToRays(fring)
    pixels = ringim.RaysToPixels(rays)

    print ""
    print "Line, sample..."
    print pixels

    print ""
    print "Selected line, sample..."
    print pixels[np.where(ringim.isInside(pixels, slop=100))]

    print ""
    print "Selected longitudes"
    print cyl[np.where(ringim.isInside(pixels, slop=100))]

def test2():

    ringim = RingImage("2009-043T22:46:38.791", "WAC")#, offset=(1.e-3, -2.e-3))

    print "ImageToRays..."
    rays = ringim.ImageToRays()

    print ""
    print "X, Y, Z..."
    print rays

    print ""
    print "RaysToRing..."
    rings = ringim.RaysToRing(rays)
    cyl = RingImage.Cylindrical(rings)
    cyl[:,:,1] *= 180./np.pi

    print ""
    print "Radius, longitude, elevation..."
    print cyl

    print ""
    print "ringim.incidence..."
    print ringim.incidence * 180./np.pi

    print ""
    print "ringim.RaysToEmission(rays)..."
    print ringim.RaysToEmission(rays) * 180./np.pi

    print ""
    print "ringim.RaysToMu(rays)..."
    print ringim.RaysToMu(rays)

    print ""
    print "ringim.RaysToPhase(rays)..."
    print ringim.RaysToPhase(rays) * 180./np.pi

    print ""
    print "ringim.RingResolution(rays,rings)..."
    print ringim.RingResolution(rays,rings)

# Execute the main test progam if this is not imported
if __name__ == "__main__": test2()
