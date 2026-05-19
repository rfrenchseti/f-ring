from optparse import OptionParser
import ringutil
import mosaic
import ringimage
import pickle
import os
import os.path
import fitreproj
import vicar
import numpy as np
import sys
import cspice
import scipy.ndimage.interpolation as interp
from imgdisp import ImageDisp, FloatEntry, ScrolledList
from Tkinter import *
from PIL import Image
import scipy.ndimage.filters as filters
import scipy.fftpack as fftpack
import ringimage as ri
import matplotlib.pyplot as plt

python_filename = sys.argv[0]

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['ISS_007RI_HPMRDFMOV001_PRIME/N1493852702_1',
#                '--recompute-reproject',
#                '--recompute-auto-offset',
#                '--display-offset-reproject',
                 '--verbose']
#    cmd_line = ['-a', '--verbose']
    pass

parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

WHOLE_FRING = (138500., 140220., 141940.)

NPIXELS = 1024
NAC_FOV_DEGREES = 0.35
WAC_FOV_DEGREES = 3.48

NAC_PIXEL_SIZE = 0.35 / NPIXELS * np.pi/180.
WAC_PIXEL_SIZE = 3.48 / NPIXELS * np.pi/180.

NAC_FRAME_NAME = "CASSINI_ISS_NAC"
WAC_FRAME_NAME = "CASSINI_ISS_WAC"

NAC_FRAME_ID   = cspice.bodn2c(NAC_FRAME_NAME)
WAC_FRAME_ID   = cspice.bodn2c(WAC_FRAME_NAME)

# Useful dictionaries
CAMERA_FRAME_NAME = {"NAC":NAC_FRAME_NAME, "WAC":WAC_FRAME_NAME}
CAMERA_FRAME_ID   = {"NAC":NAC_FRAME_ID,   "WAC":WAC_FRAME_ID}
CAMERA_PIXEL_SIZE = {"NAC":NAC_PIXEL_SIZE, "WAC":WAC_PIXEL_SIZE}


def correlate2d(image, model, normalize=False):
    """Correlate the image with the model; normalization to [-1., 1.] is optional.
    This is used to find the location of the epsilon ring."""
    assert image.shape == model.shape

    image_fft = fftpack.fft2(image)
    model_fft = fftpack.fft2(model)
    corr = np.real(fftpack.ifft2(image_fft * np.conj(model_fft)))
        
    if normalize:
        corr /= np.sqrt(np.sum(image**2) * np.sum(model**2))
    return corr

def find_pointing_offset_correlate(ringim,fring_pix, fakeim, limits, verbose=False, allow_rotation=False,
                                   ignore_exptime=False, debug=False):
#    limits = epsilon_limits
    
    
#    data = ringim.array[(fring_pix[:,0],fring_pix[:,1])]
#    print data.shape
#    print fring_pix.shape
    limits = (max(limits[0], -ringim.array.shape[0]//2+50),
              min(limits[1], ringim.array.shape[0]//2-50),
              max(limits[2], -ringim.array.shape[1]//2+50),
              min(limits[3], ringim.array.shape[1]//2-50))
    if verbose:
        print 'EPSILON Lim [%d:%d,%d:%d]'%limits,
#    epsilon, model = make_correlation_image(snapshot, object='epsilon')
#    if model is None:
#        print '==> Epsilon ring not in view!'
#        snapshot.fov = orig_fov
#        return offset_data

    min_u, max_u, min_v, max_v = limits

    min_u = np.clip(min_u, -ringim.array.shape[1]//2+1, ringim.array.shape[1]//2-1)    
    max_u = np.clip(max_u, -ringim.array.shape[1]//2+1, ringim.array.shape[1]//2-1)    
    min_v = np.clip(min_v, -ringim.array.shape[0]//2+1, ringim.array.shape[0]//2-1)    
    max_v = np.clip(max_v, -ringim.array.shape[0]//2+1, ringim.array.shape[0]//2-1)    

    min_u = np.clip(min_u, -(ringim.array.shape[1]-1), ringim.array.shape[1]-1)
    max_u = np.clip(max_u, -(ringim.array.shape[1]-1), ringim.array.shape[1]-1)
    min_v = np.clip(min_v, -(ringim.array.shape[0]-1), ringim.array.shape[0]-1)
    max_v = np.clip(max_v, -(ringim.array.shape[0]-1), ringim.array.shape[0]-1)

    # Locate the pixel offsets with the highest correlation between image and model
    corr = correlate2d(ringim.array, fakeim.array)
    if debug:
        plt.imshow(corr)
        plt.title('Correlation')
        plt.show()
        
    u_list = []
    v_list = []
    if min_u < 0 and max_u >= 0:
        u_list.append((max_u+1,min_u))
    else:
        u_list.append((0,min_u))
        u_list.append((max_u+1,corr.shape[1]))
    if min_v < 0 and max_v >= 0:
        v_list.append((max_v+1,min_v))
    else:
        v_list.append((0,min_v))
        v_list.append((max_v+1,corr.shape[0]))
#        print u_list, v_list
    for u_pair in u_list:
        corr[:, u_pair[0]:u_pair[1]] = -2
    for v_pair in v_list:
        corr[v_pair[0]:v_pair[1], :] = -2
    if debug:
        plt.imshow(corr)
        plt.title('Correlation Valid Range')
        plt.show()
    corr_max = corr.max()
    (vmax,umax) = np.where(corr == corr_max)        
    umax = umax[0]
    vmax = vmax[0]

    if umax > corr.shape[0]//2: umax -= corr.shape[0]
    if vmax > corr.shape[1]//2: vmax -= corr.shape[1]
    
    print umax, vmax
    return (umax, vmax)
#---------------------------------------------------------------------------------------------------------------------
for obsid, image_name, image_path in ringutil.enumerate_files(options, args, '_CALIB.IMG'):
    print image_path
    vicar_data = vicar.VicarImage.FromFile(image_path)
    ringim = fitreproj.VicarToRingImage(vicar_data)
    if ringim.array.shape != (1024,1024):
        print 'ERROR! Image is not 1024x1024 - aborting' # XXX
        
    
    time = ringim.et
    fring = fitreproj.MakeFRing(140220., time) # Make a series of radius/longitude pairs all the way around the ring
    rays = ringim.RingToRays(fring)
    pixels = ringim.RaysToPixels(rays)
    
    savearray = ringim.array # Same the image before we smooth it
    
    # Smooth the image.  This makes find an offset work better.
    smooth = filters.gaussian_filter(ringim.array, sigma=2, mode="constant")
    ringim.array = smooth
    
    #1. create fake image
    fake_arr = np.zeros(ringim.array.shape)
    fake_image = ri.RingImage(time, ringim.camera)
    fake_image.SetArray(fake_arr.astype("float"))

    restrict = np.where(ringim.isInside(pixels))
    p = (pixels[restrict] + 0.5).astype("int")
    fake_image.array[(p[:,0],p[:,1])] = 100.0     #fill with some random brightness
    
    xmax, ymax = find_pointing_offset_correlate(ringim, p, fake_image, (-100, 100, -100, 100))

    x_offset = xmax*CAMERA_PIXEL_SIZE[ringim.camera]
    y_offset = ymax*CAMERA_PIXEL_SIZE[ringim.camera]

    print x_offset, y_offset
    
    