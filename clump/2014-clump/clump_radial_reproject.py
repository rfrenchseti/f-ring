'''
Use:
Called from the clump_gui to create zoomed in radial images of clumps.
See "create_radial_mosaics.py" to create full, independent radial mosaics.

Author: S. Hicks, R. French

'''


from optparse import OptionParser
import ringutil
import pickle
import numpy as np
import sys
from imgdisp import ImageDisp, FloatEntry, ScrolledList
from Tkinter import *
import pylab
from clumputil import ClumpData, ClumpDBEntry
import clumputil
import numpy.ma as ma
import scipy as sp
import matplotlib.pyplot as plt
import Image
import math

class ClumpChainDispData:
    def __init__(self):
        self.toplevel = None
        self.imdisp_offset = None
        self.imdisp_repro = None

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = []
    
parser = OptionParser()

ringutil.add_parser_options(parser)
options, args = parser.parse_args(cmd_line)

blackpoint = 0.
whitepoint = options.global_whitepoint
gamma = 0.5
    
def calc_polar_params(im, width, center, longitudes):
    
    #degree increment for each pixel
    theta_step = (2*np.pi)/im.shape[1]
    
    longitudes = np.where(longitudes >= 0)[0] #we only want valid longitudes
    rad = np.arange(0,im.shape[0])
    
    #we need the same longitude for every pixel in one column
    long_rep = np.repeat(longitudes,im.shape[0]) 
    #we need one column of radii coords for every longitude
    radii = np.tile(rad, longitudes.shape[0])
    
    dx= radii*np.cos(-long_rep*theta_step)+center #convert to polar coordinates
    dy= radii*np.sin(-long_rep*theta_step)+center
    dx = np.round(dx).astype(int)
    dy = np.round(dy).astype(int)
    
    return (dx, dy, radii, long_rep, theta_step, center)

def draw_clumps(im, clump, options, obsid, color):
    
    radii = np.arange(im.shape[0])*options.radius_resolution + options.radius_start
    radius_center = np.where(radii == 140220)[0][0]
    
    width = (clump.fit_width_deg//2)/(360./im.shape[1]) #pixels
    height = 30 #pixels
    center = clump.g_center/(360./im.shape[1])
    l_thick = 4
    w_thick = 24
    
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
        
    return (center, radius_center)

def make_rad_reproj(im, obs_id, longitudes, clump, clip_width, options):
    #make the base image of the clump
    
    radii = np.arange(im.shape[0])*options.radius_resolution + options.radius_start
    radius_center = np.where(radii == 140220)[0][0]
    
    clump_long = clump.g_center/(360./im.shape[1])
    clump_center = (clump_long, radius_center)
    
    width = 2*im.shape[0]
    center = int(width/2)

    final_im = np.zeros((width, width))
    
    dx, dy, radii, long_rep, theta_step, center = calc_polar_params(im, width, center, longitudes)  
    final_im[dy, dx] = np.maximum(im[radii, long_rep], 0)**.5
    
    clump_center_x = np.round(clump_center[1]*np.cos(-clump_center[0]*theta_step) + center).astype(int)
    clump_center_y = np.round(clump_center[1]*np.sin(-clump_center[0]*theta_step) + center).astype(int)
    
    grey_clip = final_im[clump_center_y - clip_width//2:clump_center_y + clip_width//2, 
                           clump_center_x - clip_width//2:clump_center_x + clip_width//2]
      
    #returns the clipped clump image    
    return grey_clip
    
def make_rad_overlay(im, obs_id, longitudes, clump, clip_width, options):
    #create the colored clump overlay
    
    #pass draw clumps a blank/transparent image to draw on
    combined_img = np.zeros((im.shape[0], im.shape[1], 3))
    mode = 'RGB'
    
    color = (0.698, 0.133, 0.078 )
    clump_center = draw_clumps(combined_img, clump, options, obs_id, color)
    
    width = 2*im.shape[0]
    center = int(width/2)

    final_im = np.zeros((width, width, 3))  
    #new image should be a square a little larger than the 2*height of the original image.
    dx, dy, radii, long_rep, theta_step, center = calc_polar_params(im, width, center, longitudes)
    final_im[dy, dx, :] = np.maximum(combined_img[radii, long_rep, :], 0)**.5
    
    clump_center_x = np.round(clump_center[1]*np.cos(-clump_center[0]*theta_step) + center).astype(int)
    clump_center_y = np.round(clump_center[1]*np.sin(-clump_center[0]*theta_step) + center).astype(int)
    
    overlay_clip = final_im[clump_center_y - clip_width//2:clump_center_y + clip_width//2, 
                           clump_center_x - clip_width//2:clump_center_x + clip_width//2, :]
    
    #returns transparent overlay clip with clump drawn
    return overlay_clip

def make_master_img(chain, options):
    mode = 'RGB'
    sub_size = len(chain.clump_list)
    ncols = 3.
    nrows = int(math.ceil(float(sub_size)/ncols))
    ncols = int(ncols)
    
    clip_size = 300
    grey_master = np.zeros((nrows*clip_size,ncols*clip_size))
    overlay_master = np.zeros((nrows*clip_size,ncols*clip_size, 3))
    print sub_size, nrows, ncols
    
    #creates an image that reads from left to right, top to bottom. 
    #the number of rows depends on the number of images
    m = 0
    while m < sub_size:
        for k in range(nrows):
            for n in range(ncols):
                
                if m >= sub_size:
                    break
                
                col_start_idx = n*clip_size
                row_start_idx = k*clip_size
#                print col_start_idx, row_start_idx
                clump = chain.clump_list[m]
                
                obs_id = clump.clump_db_entry.obsid
                print obs_id
                print clump.g_center
                (mosaic_data_path, mosaic_metadata_path,
                 mosaic_large_png_path, mosaic_small_png_path) = ringutil.mosaic_paths(options, obs_id)
                        
                mosaic_img = np.load(mosaic_data_path + '.npy')
                mosaic_data_fp = open(mosaic_metadata_path, 'rb')
                mosaic_data = pickle.load(mosaic_data_fp)
                
                (longitudes, resolutions,
                image_numbers, ETs, 
                emission_angles, incidence_angles,
                phase_angles) = mosaic_data
                        
                grey_clip = make_rad_reproj(mosaic_img, obs_id, longitudes, clump, clip_size, options)
                overlay = make_rad_overlay(mosaic_img, obs_id, longitudes, clump, clip_size, options)
                
                overlay_master[row_start_idx:row_start_idx + overlay.shape[0], 
                               col_start_idx:col_start_idx + overlay.shape[1], :] = overlay
                grey_master[row_start_idx:row_start_idx + grey_clip.shape[0], 
                            col_start_idx:col_start_idx + grey_clip.shape[1]] = grey_clip
                                
                m += 1
                
                
    return (grey_master, overlay_master, nrows*clip_size, ncols*clip_size)

clumpchaindispdata =  ClumpChainDispData()

def setup_clumps_window(grey_master, overlay_master, master_xsize, master_ysize, tk_root):
    
    clumpchaindispdata.toplevel = tk_root
    clumpchaindispdata.toplevel.title('Chains')
    frame_toplevel = Frame(clumpchaindispdata.toplevel)

    clumpchaindispdata.imdisp = ImageDisp([grey_master],parent=frame_toplevel, overlay_list = [overlay_master], flip_y=False, one_zoom=False)
    
    frame_toplevel.pack()

            
def run_clump_radial_reproject(chain, options, tk_root):
    
    grey_master, overlay_master, master_ysize, master_xsize = make_master_img(chain, options)
    setup_clumps_window(grey_master, overlay_master, master_xsize, master_ysize, tk_root)
    
#comment out these lines if you're going to run through clump_gui!!!

if __name__ == '__main__':    
    clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
    clump_chains_fp = open(clump_chains_path, 'rb')
    clump_find_options2 = pickle.load(clump_chains_fp)
    clump_chain_options = pickle.load(clump_chains_fp)
    clump_db, clump_chain_list = pickle.load(clump_chains_fp)
    clump_chains_fp.close()
    
    chain = clump_chain_list[300]
    run_clump_radial_reproject(chain, options, Tk())         
    mainloop()
