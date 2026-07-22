'''
Program to test circularizing a mosaic.
Reference: http://www.janeriksolem.net/2012/02/polar-panoramas-with-python.html

Creates individual, full radial mosaics and places them in a specified folder.
To be used in conjuction with 'shannon_animation.py' if you want to make a movie.

If you want to specify a particular clump you can constrict the longitude range in "draw_clumps"
Use options 'clump-long-min/max' to specify which clumps you want to be drawn. 
The default is for all clumps above 1 sigma to be drawn (min = 0, max = 360.)

Author: Shannon Hicks

'''

import numpy as np
import numpy.ma as ma
import scipy as sp
import Image
import ringutil
from optparse import OptionParser
import sys
import pickle
from imgdisp import ImageDisp
from ringutil import MosaicData
import pylab

cmd_line = sys.argv[1:]
if len(cmd_line) == 0:
    
    cmd_line = ['-a',
                '--ignore-voyager',
#                'ISS_085RF_FMOVIE003_PRIME_1',
#                'ISS_085RF_FMOVIE003_PRIME_2',
#                'ISS_031RF_FMOVIE001_VIMS',
#                'ISS_032RF_FMOVIE001_VIMS',
#                'ISS_033RF_FMOVIE001_VIMS',
#                'ISS_036RF_FMOVIE002_VIMS',
#                'ISS_039RF_FMOVIE001_VIMS', 
                '--opath', '/home/shannon/radial_mosaics/',
#                '--mosaic-reduction-factor', '1',
#                '--draw-clumps',
#                '--plot-results'
                ]
#    cmd_line = ['-a', 
#                '--opath', '/home/shannon/clump_images/',
#                '--mosaic-reduction-factor', '1',
#                '--draw-clumps',
#                '--ignore-bad-obsids'
#                ]
#    cmd_line = ['-a', 
#                '--opath', '/home/shicks/Documents/Movie/Movie_png/'
#                ]
#    cmd_line = ['-a']
    
parser = OptionParser()

parser.add_option('--opath', dest = 'output_path', type = 'string', default = './')
parser.add_option('--format', dest = 'format', type = 'string', default = '.png')
parser.add_option('--draw-clumps', dest = 'draw_clumps', action = 'store_true', default = False)
parser.add_option('--plot-results', dest = 'plot_results', action = 'store_true', default = False)
parser.add_option('--clump-long-min', dest = 'clump_long_min', type = 'float', default = 0.0)
parser.add_option('--clump-long-max', dest = 'clump_long_max', type = 'float', default  = 360.)

ringutil.add_parser_options(parser)
options, args = parser.parse_args(cmd_line)

mosaicdata = MosaicData()

def scale_images(options, args):
    
    min_blackpoint = 1e38
    max_whitepoint = -1e38
    for obsid, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):        
        (mosaicdata.data_path, mosaicdata.metadata_path,
            mosaicdata.large_png_path, mosaicdata.small_png_path) = ringutil.mosaic_paths(options, obsid)
        
        mosaicdata.img = np.load(mosaicdata.data_path+'.npy')
        
        mosaic_metadata_fp = open(mosaicdata.metadata_path, 'rb')
        metadata = pickle.load(mosaic_metadata_fp)
        (mosaicdata.longitudes, mosaicdata.resolutions,
        mosaicdata.image_numbers, mosaicdata.ETs, 
        mosaicdata.emission_angles, mosaicdata.incidence_angles,
        mosaicdata.phase_angles) = metadata
            
        normalize_factor = ringutil.normalized_ew_factor(np.mean(mosaicdata.phase_angles),np.mean(mosaicdata.emission_angles))

        im = mosaicdata.img*normalize_factor
        cur_blackpoint = max(np.min(im), 0)
        cur_whitepoint = np.max(im)

        if min_blackpoint > cur_blackpoint:
            min_blackpoint = cur_blackpoint  
        if max_whitepoint < cur_whitepoint:
            max_whitepoint = cur_whitepoint

    return (min_blackpoint, max_whitepoint)

def draw_clumps(im, options, obsid, color, radius_start):

    radii = np.arange(im.shape[0])*options.radius_resolution + radius_start
    radius_center = np.where(radii == 140220)[0][0]
#    print radius_center
    clump_db_path, clump_chains_path = ringutil.clumpdb_paths(options)
    clump_db_fp = open(clump_db_path, 'rb')
    clump_find_options = pickle.load(clump_db_fp)
    clump_db = pickle.load(clump_db_fp)
    clump_db_fp.close()
    
#    for obsid in clump_db.keys():
    for clump in clump_db[obsid].clump_list:
        if (clump.fit_sigma > 1.0) and (options.clump_long_min < clump.longitude < options.clump_long_max):
            width = (clump.fit_width_deg//2)/(360./im.shape[1]) #pixels
            height = 10 #pixels
            center = clump.longitude/(360./im.shape[1])
            l_thick = 4
            w_thick = 24
#            print width, center, height, clump.fit_sigma
            for i in range(im.shape[2]):
                if (center + width > im.shape[1]):
                    rem = (center + width) - im.shape[1]
                    im[radius_center + height:radius_center + height + l_thick, center-width:im.shape[1], i] = color[i] #lines to the end
                    im[radius_center -height -l_thick:radius_center - height,center - width:im.shape[1], i] = color[i]
                    im[radius_center -height -l_thick: radius_center + height +l_thick ,center-width:center-width+w_thick, i] = color[i] # left boundary
                    im[radius_center +height:radius_center + height + l_thick, 0:rem + l_thick, i] = color[i] #extend top line past 0
                    im[radius_center - height -l_thick:radius_center - height + l_thick, 0:rem +l_thick, i] = color[i] # extend bottom line
                    im[radius_center - height -l_thick:radius_center + height +l_thick, rem:rem + w_thick, i] = color[i] #right boundary
                if (center - width < 0):
                    rem =  abs(center - width)
                    im[radius_center + height:radius_center + height+l_thick, 0:center + width +l_thick, i] = color[i]
                    im[radius_center - height - l_thick:radius_center - height, 0:center + width +l_thick, i] = color[i]
                    im[radius_center - height -l_thick:radius_center - height, im.shape[1] - rem -(l_thick +1):im.shape[1], i] = color[i]
                    im[radius_center + height: radius_center + height + l_thick, im.shape[1] - rem -(l_thick +1): im.shape[1], i] = color[i]
                    im[radius_center - height -l_thick:radius_center + height +l_thick, center + width: center + width + w_thick, i] = color[i]
                    im[radius_center -height -l_thick: radius_center + height +l_thick, im.shape[1] - rem -w_thick:im.shape[1] - rem, i] = color[i]
                else:
                    im[radius_center + height:radius_center + height + l_thick, center-width -l_thick:center + width +l_thick, i] = color[i]
                    im[radius_center -height - l_thick:radius_center - height,center - width -l_thick:center + width +l_thick, i] = color[i]
                    im[radius_center -height -l_thick: radius_center + height + l_thick,center-width:center-width+w_thick, i] = color[i]
                    im[radius_center - height -l_thick: radius_center + height + l_thick, center + width:center + width + w_thick, i] = color[i]
#                    print im[radius_center - height -l_thick: radius_center + height + l_thick, center + width:center + width + w_thick, i]
'''
--------------------------------------------
        MAIN PROGRAM
--------------------------------------------
'''

  #SCALE IMAGES APPROPRIATELY
#blackpoint,whitepoint = scale_images(options, args)
blackpoint = 0.
whitepoint = options.global_whitepoint
gamma = 0.5

for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
    
    (mosaicdata.data_path, mosaicdata.metadata_path,
     mosaicdata.large_png_path, mosaicdata.small_png_path) = ringutil.mosaic_paths(options, obs_id)
     
    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
     bkgnd_mask_filename, bkgnd_model_filename,
     bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)
            
    if options.mosaic_reduction_factor != 1:        
        mosaic_img = np.load(reduced_mosaic_data_filename+'.npy')
    
        mosaic_img = mosaic_img.view(ma.MaskedArray)
        mosaic_img.mask = np.load(bkgnd_mask_filename+'.npy')
                
        reduced_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
        mosaic_data = pickle.load(reduced_metadata_fp)
        (mosaicdata.longitudes, mosaicdata.resolutions,
        mosaicdata.image_numbers, mosaicdata.ETs, 
        mosaicdata.emission_angles, mosaicdata.incidence_angles,
        mosaicdata.phase_angles) = mosaic_data
    else:
        mosaic_img = np.load(mosaicdata.data_path + '.npy')
        mosaic_data_fp = open(mosaicdata.metadata_path, 'rb')
        mosaic_data = pickle.load(mosaic_data_fp)
        (mosaicdata.longitudes, mosaicdata.resolutions,
        mosaicdata.image_numbers, mosaicdata.ETs, 
        mosaicdata.emission_angles, mosaicdata.incidence_angles,
        mosaicdata.phase_angles) = mosaic_data
        
    print 'OBS_ID: ', obs_id
    im = mosaic_img
    
    im = im[250:650, :]
    radius_start = options.radius_start + 250*options.radius_resolution
    
    combined_img = np.zeros((im.shape[0], im.shape[1], 3))
    combined_img[:, :, 0] = im
    combined_img[:, :, 1] = im
    combined_img[:, :, 2] = im
    mode = 'RGB'
    
    if options.draw_clumps:
        color = (255,0,0)
#        color = (0.69, 0.13, 0.07 )
        draw_clumps(combined_img, options, obs_id, color, radius_start)
    
    #new image should be a square a little larger than the 2*height of the original image.
    
    width = 2*im.shape[0]
    center = int(width/2)

    final_im = np.zeros((width, width, 3))
 
    theta_step = (2*np.pi)/im.shape[1]
    theta_array = np.arange(0, 2*np.pi, theta_step)
    
    longitudes = np.where(mosaicdata.longitudes >= 0)[0]
    rad = np.arange(0,im.shape[0])
    long_rep = np.repeat(longitudes,im.shape[0])
    radii = np.tile(rad, longitudes.shape[0])
    
    dx= radii*np.cos(long_rep*theta_step)+center
    dy= radii*np.sin(long_rep*theta_step)+center
    dx = np.round(dx).astype(int)
    dy = np.round(dy).astype(int)

    for i in range(final_im.shape[2]):
#        print i
        final_im[dy, dx, i] = np.maximum(combined_img[radii, long_rep, i], 0)**.5

    final_im = ImageDisp.ScaleImage(final_im, blackpoint, whitepoint, gamma)[::-1,:]+0
    final_im = np.cast['int8'](final_im)
    final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]),
                           final_im, 'raw', mode, 0, 1)
    
    if options.plot_results:
        pylab.imshow(final_img)
        pylab.show()
    
    sp.misc.imsave(options.output_path + obs_id + options.format, final_img)    
