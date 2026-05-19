import matplotlib.pyplot as plt
import numpy as np
import pickle
from optparse import OptionParser
import ringutil
import clumputil
import sys
import os
import numpy.ma as ma
from imgdisp import ImageDisp
import Image
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import cspice
import matplotlib

cmd_line = sys.argv[1:]
if len(cmd_line) == 0:
    
    cmd_line = ['-a']
    
parser = OptionParser()    
ringutil.add_parser_options(parser)
options, args = parser.parse_args(cmd_line)

radius_res = options.radius_resolution
radius_start = options.radius_start
blackpoint = 0.0
gamma = 0.5
color_foreground = 'black'

matplotlib.rc('xtick', color=color_foreground, labelsize=7)
matplotlib.rc('xtick.major', size=6)
matplotlib.rc('xtick.minor', size=4)
matplotlib.rc('ytick', color=color_foreground, labelsize=7)
matplotlib.rc('ytick.major', size=6)
matplotlib.rc('ytick.minor', size=4)
matplotlib.rc('font', size=7)
matplotlib.rc('legend', fontsize=7)


def make_mosaic_clip(obsid, clump_db, mosaic_dimensions):
    
    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
                bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obsid)
                        
    mosaic_img = np.load(reduced_mosaic_data_filename + '.npy')
    mosaic_data_fp = open(reduced_mosaic_metadata_filename, 'rb')
    mosaic_data = pickle.load(mosaic_data_fp)
    
    (longitudes, resolutions,
    image_numbers, ETs, 
    emission_angles, incidence_angles,
    phase_angles) = mosaic_data
    
    mu = ringutil.mu(c_approved_db[obsid].emission_angle)
    mosaic_img = mosaic_img*mu
#            print mosaic_img.shape[0], mosaic_img.shape[1]
#        mosaic_max.append(ma.max(mosaic_img[400:650, (x_min_left -10.)/long_res: (x_max_right +10.)/long_res + 1]))
    mosaic_clip = mosaic_img[mosaic_dimensions[0]:mosaic_dimensions[1], mosaic_dimensions[2]:mosaic_dimensions[3]]
#            print mosaic_clip.shape[0], mosaic_clip.shape[1]
    
    color_mosaic = np.zeros((mosaic_clip.shape[0], mosaic_clip.shape[1], 3))
    color_mosaic[:,:,0] = mosaic_clip
    color_mosaic[:,:,1] = mosaic_clip
    color_mosaic[:,:,2] = mosaic_clip
    
    return color_mosaic

def edit_mosaic_axis(m_ax):
    
    m_ax.xaxis.tick_bottom()
    plt.setp(m_ax.get_xticklabels(), visible=False)
    plt.setp(m_ax.get_yticklabels(), visible=False)
    m_ax.tick_params(axis = 'x', direction = 'out', length = 2.0)
    m_ax.xaxis.tick_bottom()
    m_ax.set_yticks([])
    
def edit_profile_axis(p_ax):
    
    yFormatter = FormatStrFormatter('%.1f')
    xFormatter = FormatStrFormatter('%d')
    p_ax.yaxis.set_major_formatter(yFormatter)
    p_ax.xaxis.set_major_formatter(xFormatter)
    p_ax.yaxis.tick_left()
    p_ax.xaxis.tick_bottom()
    plt.setp(p_ax.get_xticklabels(), visible=False)
    
    
def plot_single_ew_profile(ax, ew_data, clump_db_entry, long_min, long_max, label=False, color='black'):
    
    long_res = 360. / len(ew_data)
    longitudes = np.arange(len(ew_data)) * long_res
    min_idx = int(long_min / long_res)
    max_idx = int(long_max / long_res)
    long_range = longitudes[min_idx:max_idx]
    ew_range = ew_data[min_idx:max_idx]
    legend = None
    if label:
        legend = clump_db_entry.obsid + ' (' + cspice.et2utc(clump_db_entry.et, 'C', 0) + ')'
    ax.plot(long_range, ew_range, '-', label=legend, color=color)
    
def plot_clumps(c_approved_list, c_approved_db):
    
    
    def draw_clumps(im, clump, color, mosaic_dimensions,rad_center = 140220.):
        top_row, bot_row, left_bound, right_bound = mosaic_dimensions
        long_res = 360./len(clump.clump_db_entry.ew_data)
        radii = np.arange(len(clump.clump_db_entry.ew_data))*radius_res + radius_start
        radii = radii[top_row:bot_row]
        radius_center = np.where(radii == rad_center)[0][0]
#        print radius_res, im.shape[1], radius_center
#        sys.exit(2)
        
        left_idx = (clump.fit_left_deg)/(long_res)-left_bound #pixels
        right_idx = (clump.fit_right_deg)/(long_res) - left_bound
        height = 30 #pixels
#        center = clump.g_center/(360./im.shape[1])
        l_thick = 4
        w_thick = 4
        
        for i in range(len(color)):
            im[radius_center + height:radius_center + height + l_thick, left_idx:right_idx +l_thick, i] = color[i]
            im[radius_center - height - l_thick:radius_center - height, left_idx:right_idx +l_thick, i] = color[i]
            im[radius_center - height - l_thick: radius_center + height + l_thick,left_idx:left_idx+w_thick, i] = color[i]
            im[radius_center - height - l_thick: radius_center + height + l_thick, right_idx:right_idx + w_thick, i] = color[i]
        
    chain_time_db = {}
    for chain in c_approved_list:
        chain.skip = False
        start_date = chain.clump_list[0].clump_db_entry.et_max
        if start_date not in chain_time_db.keys():
            chain_time_db[start_date] = []
            chain_time_db[start_date].append(chain)
        elif start_date in chain_time_db.keys():
            chain_time_db[start_date].append(chain)
    
    for obsid in c_approved_db:
        max_time = c_approved_db[obsid].et_max
        if max_time not in chain_time_db.keys():
            chain_time_db[max_time] = []
            
    for chain_time in chain_time_db:
        chain_list = chain_time_db[chain_time]
        chain_list.sort(key=lambda x: x.clump_list[0].g_center * 1000 + x.clump_list[1].g_center)
            
    num = 1        
    for time in sorted(chain_time_db.keys()):
        for a,chain in enumerate(chain_time_db[time]):
            if chain.skip == False:
                parent_clump_start_long = '%6.2f'%(chain.clump_list[0].g_center)
                parent_clump_end_long = '%6.2f'%(chain.clump_list[-1].g_center)
                parent_clump_end_time = chain.clump_list[-1].clump_db_entry.et_max
                num_id = 'C'+str(num)
                found_start_split = False
                found_end_split = False
                #check to see if this clump is the beginning of a split
                for b, new_chain in enumerate(chain_time_db[time][a+1::]):
                    new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
#                    print parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                    if new_parent_start_long == parent_clump_start_long:
                        print 'Found a splitting clump', parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                        new_num_id = num_id+"'"
                        
                        #skip this clump so that it isn't put in the table a second time
                        new_chain.skip = True
                        
                        split_db = {}
                        x_min_left = 9999
                        x_max_right = -9999
                        for clump in chain.clump_list:
                            if clump.fit_left_deg < x_min_left:
                                x_min_left = clump.fit_left_deg
                            if clump.fit_right_deg > x_max_right:
                                x_max_right = clump.fit_right_deg
                            t = clump.clump_db_entry.et_max
                            
                            if t not in split_db.keys():
                                split_db[t] = []
                                split_db[t].append((num_id,clump))
                            else:
                                split_db[t].append((num_id,clump))
                                
                        for clump in new_chain.clump_list:
                            if clump.fit_left_deg < x_min_left:
                                x_min_left = clump.fit_left_deg
                            if clump.fit_right_deg > x_max_right:
                                x_max_right = clump.fit_right_deg
                            t = clump.clump_db_entry.et_max
                            
                            if t not in split_db.keys():
                                split_db[t] = []
                                split_db[t].append((new_num_id,clump))
                            else:
                                split_db[t].append((new_num_id, clump)) 
                        
                        #round xmax and xmin to nearest multiple of five for graphing prettiness
                        x_min_left = np.floor((x_min_left/5.))*5.
                        x_max_right = np.ceil((x_max_right/5.))*5.
                        fig = plt.figure(figsize = (7.0,3.0))
                        num_subplots = len(split_db.keys())
                        
                        
                        ax_num = 1
                        mosaic_max = []
                        mosaic_clips = []
                        clump_data_db = {}
                        for key in split_db.keys():
                            clump_data_db[key] = []
                                
                        for d, key in enumerate(sorted(split_db.keys())):
                            
                            ax = fig.add_subplot(num_subplots, 2, ax_num)
                            ew_data = split_db[key][0][1].clump_db_entry.ew_data
                           
                            long_res = 360./len(ew_data)
                            mosaic_dimensions = (400, 650, (x_min_left-10.)/long_res, (x_max_right +10.)/long_res)
                            
                            
                            ax.get_xaxis().set_ticks(np.arange(x_min_left-10, x_max_right +10.+ 10, 10))
                            ax.set_xlim(x_min_left-10., x_max_right +10.)
                            y_min = np.min(ew_data[(x_min_left-10.)/long_res: (x_max_right + 10.)/long_res])
                            y_max = np.max(ew_data[(x_min_left-10.)/long_res: (x_max_right + 10.)/long_res]) - y_min +0.2
                            ystep = (y_max)/4.
                            ax.get_yaxis().set_ticks(np.arange(0.0, y_max + ystep, ystep ))
                            ax.set_ylim(0., y_max) 
                            edit_profile_axis(ax)
                            
#                            ax.text(0.37, 0.87, cspice.et2utc(split_db[key][0][1].clump_db_entry.et_min, 'D',5)[:20], transform = ax.transAxes)
                            ax.text(0.37, 0.87, split_db[key][0][1].clump_db_entry.obsid, transform = ax.transAxes)
                            
                            plot_single_ew_profile(ax, ew_data-y_min, split_db[key][0][1].clump_db_entry, 0., 360., color = 'black')
                            
                        #plot image of splitting clump
                            ax2 = fig.add_subplot(num_subplots, 2, ax_num +1)
                            edit_mosaic_axis(ax2)
                            
                            obsid = split_db[key][0][1].clump_db_entry.obsid
                            xticks = np.arange((x_min_left-10),x_max_right+10 + 10., 10.)*(1/long_res)
                            ax2.set_xticks(xticks - xticks[0])
                            xtick_labels = xticks*long_res
                            plt.setp(ax2, 'xticklabels', [str(int(tick)) for tick in xtick_labels] )  #for some reason ax2.set_xticklabels doesn't work - so we do it this way
                           
                            mosaic_clip = make_mosaic_clip(obsid,c_approved_db, mosaic_dimensions)                            
                            mosaic_max.append( ma.max(mosaic_clip))    
                            for i,clump_data in enumerate(split_db[key]):
                                
                                clump_id, clump = clump_data
                              
                                colors = ['#A60C00', '#0AAAC2']
                                
#                                if i == 0:
                                if (d==0) and (i==0):
                                    ax.text(clump.g_center-0.5, y_max-0.15, clump_id)
                                    clumputil.plot_fitted_clump_on_ew(ax,clump.clump_db_entry.ew_data-y_min, clump, color = 'black')
                                    clump_data_db[key].append((clump, d, i, 1))
                                if (d !=0):
                                    clump_id = clump_id + "'"
                                    print clump_id
                                    if clump_id[-2::] == "''":
                                        color_clump = colors[-1]
                                        clump_data_db[key].append((clump, d, i, -1))
                                    else:
                                        color_clump = colors[0]
                                        clump_data_db[key].append((clump, d, i, 0))
                                        
                                    clumputil.plot_fitted_clump_on_ew(ax,clump.clump_db_entry.ew_data-y_min, clump, color = color_clump)
#                                    draw_clumps(color_mosaic, clump, im_clump_color)
                                    ax.text(clump.g_center-1.0, y_max-0.15, clump_id)
                            
                            
                            mosaic_clips.append(mosaic_clip)
                            ax_num += 2
                            
                        print clump_data_db
                        mosaic_max = ma.max(mosaic_max)
                        even_axes = range(len(fig.axes))[1::2]
                        et_keys = sorted(split_db.keys())
                        for l, ax_num in enumerate(even_axes):
                            rgb_colors = [(166,12,0),(10,170,194)] #same as colors, but in rgb form
                            ax2 = fig.axes[ax_num]
                            mosaic_clip = mosaic_clips[l]
                            mode = 'RGB'
                            final_im = ImageDisp.ScaleImage(mosaic_clip, blackpoint, mosaic_max, gamma)+0
                            
                            
                            for clump_data in clump_data_db[et_keys[l]]:
                                clump, row, clump_num, color_key = clump_data
                                if (row == 0) and (clump_num == 0):
                                    im_clump_color = (255, 255,255)
                                    draw_clumps(final_im, clump, im_clump_color, mosaic_dimensions)
                                if (row != 0):
                                    draw_clumps(final_im, clump, rgb_colors[color_key], mosaic_dimensions)
                                    
                            final_im = np.cast['int8'](final_im)
                            final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]),
                                                   final_im, 'raw', mode, 0, 1)
                            ax2.imshow(final_img, aspect = 'auto')
                        
                        fig.tight_layout()
                        fig.subplots_adjust(hspace = 0.1, wspace = 0.08)
#                        plt.setp(ax.get_xticklabels(), visible=True)
                        plt.setp(ax2.get_xticklabels(), visible=True)
                        ax2.tick_params(axis = 'x', direction = 'in', length = 2.0)
#                        plt.setp(ax.get_yticklabels(), visible = True)
                        
                        ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 7)
                        plt.setp(ax.get_xticklabels(), visible=True)
                        plt.figtext(-0.01, .72, 'Normalized Equivalent Width (km)', rotation = 'vertical', fontsize = 7)
                        plt.savefig(os.path.join(ringutil.ROOT, 'chain_images','chain_' + num_id +'.png'), bbox_inches='tight', facecolor='white', dpi=500) 
                        
                        found_start_split = True 
#                        plt.show()    

                #check to see if parent chain split at the end
                split_chains = []
                c = 0
                for new_chain in chain_time_db[parent_clump_end_time]:
                    new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
                    if new_parent_start_long == parent_clump_end_long:
                        print 'Parent clump has split', parent_clump_end_long, chain.clump_list[-1].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                        new_num_id = num_id + "'"*(c+1)
                        print new_num_id
                        print '%6.2f'%(new_chain.clump_list[1].g_center) 
                        print len(new_chain.clump_list)
                        if num_id == 'C61':
                            if len(new_chain.clump_list) == 2:
                                new_num_id = num_id + "'"
                                print new_num_id
                            if len(new_chain.clump_list) > 2:
                                if ('%6.2f'%(new_chain.clump_list[1].g_center) == '214.29') and ('%6.2f'%(new_chain.clump_list[2].g_center) == '209.36'):
                                    new_num_id = num_id + "'''"
                                    print new_num_id
                                if ('%6.2f'%(new_chain.clump_list[1].g_center) == '222.44'): 
                                    new_num_id = num_id + "''"
                                    print new_num_id
                                if ('%6.2f'%(new_chain.clump_list[1].g_center) == '214.29') and ('%6.2f'%(new_chain.clump_list[2].g_center) == '212.81'):
                                    new_num_id = num_id + "''''"
                                    print new_num_id
                        if num_id == 'C35':
                            if ('%6.2f'%(new_chain.clump_list[1].g_center) == ' 90.79'):
                                new_num_id = num_id + "'"
                            if ('%6.2f'%(new_chain.clump_list[1].g_center) == ' 97.96'):
                                new_num_id = num_id + "''"
                            if ('%6.2f'%(new_chain.clump_list[1].g_center) == '104.76'):
                                new_num_id = num_id + "'''"
                        split_chains.append((new_num_id,new_chain))
                        #skip the chain so that it isn't put in the table a second time
                        new_chain.skip = True
                        c +=1
                             
                    if len(split_chains) == 2 or (len(split_chains) ==3) or (len(split_chains) == 4):
                        split_db = {}
                        x_max_right = -9999
                        x_min_left = 9999
                        for clump_id, split_chain in split_chains:
                            for k,clump in enumerate(split_chain.clump_list):
                                if clump.fit_left_deg < x_min_left:
                                    x_min_left = clump.fit_left_deg
                                if clump.fit_right_deg > x_max_right:
                                    x_max_right = clump.fit_right_deg
                                t = clump.clump_db_entry.et_max
                                if t not in split_db.keys():
                                    split_db[t] = []
                                    if k ==0:
                                        split_db[t].append((num_id, clump))
                                    else:
                                        split_db[t].append((clump_id, clump))
                                else:
                                    if k ==0:
                                        split_db[t].append((num_id, clump))
                                    else:
                                        split_db[t].append((clump_id, clump))
                        
                        fig = plt.figure(figsize = (7.0,7))
                        x_min_left = np.floor((x_min_left/5.))*5.
                        x_max_right = np.ceil((x_max_right/5.))*5.
                        parent_axes = len(chain.clump_list[:-1])
                        ax_num = 1
                        mosaic_max = []
                        mosaic_clips = []
                        clump_data_db = {}
                        for idx in range(len(chain.clump_list[:-1])):
                            clump_data_db[idx] = []
                            
                            
                        for i,clump in enumerate(chain.clump_list[:-1]):
                            
                            clump_data_db[i].append((clump, 10))
                            ax = fig.add_subplot(parent_axes, 2, ax_num)
                            ew_data = clump.clump_db_entry.ew_data
                            
                            long_res = 360./len(ew_data)
                            mosaic_dimensions = (400, 650, (x_min_left-10.)/long_res, (x_max_right +10.)/long_res)
                            ax.get_xaxis().set_ticks(np.arange(x_min_left-10., x_max_right +10. + 10, 10))
                            ax.set_xlim(x_min_left-10., x_max_right+10.)
                            y_min = np.min(ew_data[(x_min_left-10.)/long_res: (x_max_right + 10.)/long_res])
                            y_max = np.max(ew_data[(x_min_left-10.)/long_res: (x_max_right + 10.)/long_res]) - y_min
                            if num_id == 'C35':
                                y_max += 0.2
                            if num_id == 'C61':
                                y_max += 0.5
                            ystep = (y_max)/4.
                            ax.get_yaxis().set_ticks(np.arange(0.0, y_max + ystep, ystep ))
                            ax.set_ylim(0.0, y_max) 
                            edit_profile_axis(ax)
                            
                            y_clump_max = np.max(ew_data[clump.fit_left_deg/long_res:clump.fit_right_deg/long_res])- y_min
                            ax.text(clump.g_center -1.0, y_clump_max + 0.08, num_id)
                            
                            plot_single_ew_profile(ax, ew_data-y_min, clump.clump_db_entry, 0., 360., color = 'black')
                            clumputil.plot_fitted_clump_on_ew(ax, ew_data-y_min, clump, color = 'black')
                            
#                            ax.text(0.37, 0.87, cspice.et2utc(clump.clump_db_entry.et_min, 'D',5)[:20], transform = ax.transAxes)
                            ax.text(0.37, 0.87, clump.clump_db_entry.obsid, transform = ax.transAxes)

                            #plot image------------------------------------
                            ax2 = fig.add_subplot(parent_axes, 2, ax_num +1)
                            obsid = clump.clump_db_entry.obsid
                            xticks = np.arange((x_min_left-10),x_max_right+10 + 10., 10.)*(1/long_res)
                            ax2.set_xticks(xticks - xticks[0])
                            xtick_labels = xticks*long_res
                            plt.setp(ax2, 'xticklabels', [str(int(tick)) for tick in xtick_labels] )  #for some reason ax2.set_xticklabels doesn't work - so we do it this way
                            ax2.tick_params(axis = 'x', direction = 'out', length = 2.0)
                            edit_mosaic_axis(ax2)
                            
                            mosaic_clip = make_mosaic_clip(obsid, c_approved_db, mosaic_dimensions)
                            mosaic_max.append(ma.max(mosaic_clip))
                            mosaic_clips.append(mosaic_clip)
                            ax_num += 2
                            
                        num_subplots = len(split_db.keys())
                        #change the axes geometry to add the split chains to the figure
                        total_subs = num_subplots + parent_axes
                        for m in range(len(fig.axes)):
                            fig.axes[m].change_geometry(total_subs, 2, m+1)
                        start_ax = len(fig.axes)
                        ax_num = start_ax +1
                        
                        print clump_data_db
                        offset = len(clump_data_db.keys())
                        for key in range(len(split_db.keys())):
                            clump_data_db[key + offset] = []
                        print clump_data_db   
                        for d, key in enumerate(sorted(split_db.keys())):
                            sub_num = parent_axes + d+1 
                            ew_data = split_db[key][0][1].clump_db_entry.ew_data
                            long_res = 360./len(ew_data)
                            mosaic_dimensions = (400, 650, (x_min_left-10.)/long_res, (x_max_right +10.)/long_res)
                            
                            ax = fig.add_subplot(total_subs, 2, ax_num)
                            ax.get_xaxis().set_ticks(np.arange(x_min_left-10.,x_max_right +10.+10, 10))
                            ax.set_xlim(x_min_left - 10., x_max_right + 10.)
                            y_min = np.min(ew_data[(x_min_left-10.)/long_res:(x_max_right-10.)/long_res])
                            y_max = np.max(ew_data[(x_min_left-10.)/long_res:(x_max_right-10.)/long_res]) - y_min
                            if (d==0) and (num_id == 'C35'):
                                y_max += 0.2
                            if (d==0) and (num_id == 'C61'):
                                y_max += 0.5
                            elif d !=0:
                                y_max += 0.2
                            ystep = y_max/4.
                            ax.get_yaxis().set_ticks(np.arange(0.0, y_max + ystep, ystep ))
                            ax.set_ylim(0.0, y_max)
                            ax.yaxis.tick_left()
                            ax.xaxis.tick_bottom()
                            plt.setp(ax.get_xticklabels(), visible=False)
                            
#                            ax.text(0.37, 0.87, cspice.et2utc(split_db[key][0][1].clump_db_entry.et_min, 'D',5)[:20], transform = ax.transAxes)
                            ax.text(0.37, 0.87, split_db[key][0][1].clump_db_entry.obsid, transform = ax.transAxes)
                            plot_single_ew_profile(ax, ew_data- y_min, split_db[key][0][1].clump_db_entry, 0., 360., color = 'black')
                            
                            ax2 = fig.add_subplot(total_subs, 2, ax_num +1)
                            obsid = split_db[key][0][1].clump_db_entry.obsid
                            xticks = np.arange((x_min_left-10),x_max_right+10 + 10., 10.)*(1/long_res)
                            ax2.set_xticks(xticks - xticks[0])
                            xtick_labels = xticks*long_res
                            plt.setp(ax2, 'xticklabels', [str(int(tick)) for tick in xtick_labels] )  #for some reason ax2.set_xticklabels doesn't work - so we do it this way
                            ax2.tick_params(axis = 'x', direction = 'out', length = 2.0)
                            edit_mosaic_axis(ax2)
                            
                            mosaic_clip = make_mosaic_clip(obsid, c_approved_db, mosaic_dimensions)
                            mosaic_max.append(ma.max(mosaic_clip))
                            mosaic_clips.append(mosaic_clip)
                            
                            
                            for i,clump_data in enumerate(split_db[key]):
                                clump_id, clump = clump_data
#                                print i + offset, clump_id
                                if (d==1) and ((clump_id == "C61'''") or (clump_id == "C61''''")):
                                    continue
#                                print clump_id
                                colors = ['#A60C00', '#0AAAC2', '#14A300', '#9462BF']
                                rgb_colors = [(166, 12, 0), (10, 170, 194), (20, 163, 0), (197, 131,255 )]
                                if (clump_id[-4::] == "''''"):
                                    clump_color = colors[3]
                                    clump_data_db[d + offset].append((clump, 3))
#                                    im_clump_color = rgb_colors[3]
                                if (clump_id[-3::] == "'''") and (clump_id[-4::] != "''''"):
                                    clump_color = colors[2]
                                    clump_data_db[d +offset].append((clump, 2))
#                                    im_clump_color = rgb_colors[2]
                                if (clump_id[-2::] == "''") and (clump_id[-3::] != "'''") and (clump_id[-4::] != "''''"):
                                    clump_color = colors[1]
                                    clump_data_db[d +offset].append((clump, 1))
#                                    im_clump_color = rgb_colors[1]
                                if (d==0):
                                    clump_color = 'black'
                                    clump_data_db[d +offset].append((clump, 10))
                                elif (d != 0) and (clump_id[-1::] == "'") and (clump_id[-2::] != "''") and (clump_id[-3::] != "'''") and (clump_id[-4::] != "''''"):
                                    clump_color = colors[0]
#                                    im_clump_color = rgb_colors[0]
                                    clump_data_db[d +offset].append((clump, 0))
#                                draw_clumps(color_mosaic, clump, im_clump_color)
                                clumputil.plot_fitted_clump_on_ew(ax,clump.clump_db_entry.ew_data - y_min, clump, color = clump_color)
                                y_clump_max = np.max(ew_data[clump.fit_left_deg/long_res:clump.fit_right_deg/long_res]) -y_min
                                if clump_id == 'C61':
                                    y_clump_max += 0.08
                                if clump_id == 'C35':
                                    y_clump_max += 0.05
                                elif (clump_id != 'C35') and (clump_id != 'C61'):
                                    y_clump_max += 0.05
                                ax.text(clump.g_center-1.5, y_clump_max, clump_id)
#                                print clump_data_db
                           
#                                line_height = np.arange(y_min,y_max,0.01)
                        
                            xFormatter = FormatStrFormatter('%d')
                            yFormatter = FormatStrFormatter('%.1f')
                            ax.yaxis.set_major_formatter(yFormatter)    
                            ax.xaxis.set_major_formatter(xFormatter) 
                            ax_num += 2
                           
                        mosaic_max = ma.max(mosaic_max)
                        even_axes = range(len(fig.axes))[1::2]
#                        print clump_data_db
                        for l, ax_num in enumerate(even_axes):
                            rgb_colors = [(166, 12, 0), (10, 170, 194), (20, 163, 0), (197, 131,255 )] #same as clump colors for profiles
                            ax2 = fig.axes[ax_num]
                            mosaic_clip = mosaic_clips[l]
                            mode = 'RGB'
                            final_im = ImageDisp.ScaleImage(mosaic_clip, blackpoint, mosaic_max, gamma)+0
                            
                            for clump_data in clump_data_db[l]:
                                clump, color_key = clump_data
                                if color_key == 10:
                                    im_clump_color = (255, 255, 255)
                                    draw_clumps(final_im, clump, im_clump_color, mosaic_dimensions)
                                else:
                                    draw_clumps(final_im, clump, rgb_colors[color_key], mosaic_dimensions)
                                
                            final_im = np.cast['int8'](final_im)
                            final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]),
                                                   final_im, 'raw', mode, 0, 1)
                            
                            ax2.imshow(final_img, aspect = 'auto')   
                            
                        fig.tight_layout()
                        fig.subplots_adjust(hspace = 0.125, wspace = 0.08)
#                        plt.setp(ax.get_xticklabels(), visible=True)
                        plt.setp(ax2.get_xticklabels(), visible=True)
                        ax2.tick_params(axis = 'x', direction = 'in', length = 2.0)
#                        fig.subplots_adjust(hspace = 0.125)
                        plt.setp(ax.get_xticklabels(), visible=True)
                        ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 10)
                        plt.figtext(-0.025, .62, 'Normalized Equivalent Width (km)', rotation = 'vertical', fontsize = 10)

                        plt.savefig(os.path.join(ringutil.ROOT, 'chain_images','chain_' + num_id +'.png'), bbox_inches='tight', facecolor='white', dpi=1000)        
                        
                        found_end_split = True
#               

         
                if (found_end_split == False) and (found_start_split == False):
                    
                    
                    ax_num = 1
                    total_rows = len(chain.clump_list)
                    lefts = [clump.fit_left_deg for clump in chain.clump_list]
                    rights = [clump.fit_right_deg for clump in chain.clump_list]
                    min_left = np.min(lefts)
                    x_min_left = np.floor(min_left/5.)*5. -10.
                    if x_min_left < 0:
                        x_min_left = 0.
                    max_right = np.max(rights)
                    x_max_right = np.ceil(max_right/5.)*5. +10.
                    if x_max_right > 360.:
                        x_max_right = 360.
                    
                    
                    mosaic_max = []
                    mosaic_clips = []
                    y_abs_max = []
                    print x_min_left, x_max_right
                    if x_min_left < x_max_right:
                        print 'PLOTTING ', num_id
                        fig = plt.figure(figsize =(7.0,7.0))
                    #now we can plot a regular clump
                        for clump in chain.clump_list:
                            print 'CLUMP INFO-----', clump.clump_db_entry.obsid
                            print 'CLUMP LONG:', clump.g_center, 'CLUMP LEFT DEG', clump.fit_left_deg, 'CLUMP RIGHT DEG', clump.fit_right_deg
                            long_res = 360./len(clump.clump_db_entry.ew_data)
                            mosaic_dimensions = (400, 650, x_min_left/long_res, x_max_right/long_res + 1)
                            ax = fig.add_subplot(total_rows, 2, ax_num)
                        
                            ymin = np.min(clump.clump_db_entry.ew_data[x_min_left/long_res: x_max_right/long_res])
                            ymax = np.max(clump.clump_db_entry.ew_data[x_min_left/long_res: x_max_right/long_res])-ymin
                            y_abs_max.append(ymax)        
                            ystep = (ymax)/4.
                            
                    #        time = cspice.et2utc(clump_db[before_id].et_min, 'C', 0)[:11]
#                            ax.text(0.37, 0.87, cspice.et2utc(clump.clump_db_entry.et_min, 'D',5)[:20], transform = ax.transAxes)
                            ax.text(0.37, 0.87, clump.clump_db_entry.obsid, transform = ax.transAxes)
                            
                            ax.get_yaxis().set_ticks(np.arange(0.0, ymax + ystep, ystep ))
                            ax.set_ylim(0.0, ymax)
                            edit_profile_axis(ax)
                #            ax.set_yticklabels([])
                            
                            ax.get_xaxis().set_ticks(np.arange(x_min_left, x_max_right + 10, 10))
                            ax.set_xlim(x_min_left, x_max_right)
                            #plot profile
                            longitudes = np.arange(0, 360., long_res)
                            ax.plot(longitudes, clump.clump_db_entry.ew_data - ymin, color = 'black', lw = 1.5)
                            clumputil.plot_fitted_clump_on_ew(ax, clump.clump_db_entry.ew_data - ymin, clump)
                            
                            ax2 = fig.add_subplot(total_rows, 2, ax_num +1)
                            
                            xticks = np.arange(x_min_left,x_max_right + 10., 10.)*(1./long_res)
                            
                            ax2.set_xticks(xticks - xticks[0])
                            xtick_labels = xticks*long_res
                #            print xtick_labels
                            plt.setp(ax2, 'xticklabels', [str(int(tick)) for tick in xtick_labels] )  #for some reason ax2.set_xticklabels doesn't work - so we do it this way
                            edit_mosaic_axis(ax2)
                            #plot mosaic
                #            print 'MAKE MOSAIC'
                            obsid = clump.clump_db_entry.obsid
                #            print obsid
                           #            print mosaic_clip.shape[0], mosaic_clip.shape[1]
                            mosaic_clip = make_mosaic_clip(obsid, c_approved_db, mosaic_dimensions) 
                            mosaic_max.append(ma.max(mosaic_clip))
                           
                            mosaic_clips.append(mosaic_clip)
                            
                            ax_num += 2
                
                        y_abs_max = ma.max(y_abs_max)
                        #        y_abs_min = np.min(y_abs_min)
                        total_axes = len(fig.axes)
                        odd_axes = range(total_axes)[::2] #get all odd axes that have the EW data plotted
                        #        print odd_axes, total_axes
                        for ax_num in odd_axes:
                            ax = fig.axes[ax_num] 
                            ystep = (y_abs_max)/4.
                            ax.get_yaxis().set_ticks(np.arange(0.0, y_abs_max + ystep, ystep ))
                            ax.set_ylim(0.0, y_abs_max)
                            
                        #rescale all of the mosaics
                        mosaic_max = ma.max(mosaic_max)
                        even_axes = range(total_axes)[1::2]
                        for l, ax_num in enumerate(even_axes):
                            ax2 = fig.axes[ax_num]
                            mosaic_clip = mosaic_clips[l]
                #            print mosaic_clip.shape[0], mosaic_clip.shape[1], ma.count(mosaic_clip)
                            mode = 'RGB'
                            final_im = ImageDisp.ScaleImage(mosaic_clip, 0.0, mosaic_max*0.25, 0.5)+0
                            
                            draw_clumps(final_im, chain.clump_list[l], (255, 0, 0), mosaic_dimensions)
                            final_im = np.cast['int8'](final_im)
                            final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]), final_im, 'raw', mode, 0, 1)
                            
                            ax2.imshow(final_img, cmap = 'gray', aspect = 'auto')
                            
                        fig.tight_layout()
                        fig.subplots_adjust(hspace = 0.1, wspace = 0.08)
                        plt.setp(ax.get_xticklabels(), visible=True)
                        plt.setp(ax2.get_xticklabels(), visible=True)
                        plt.setp(ax.get_yticklabels(), visible = True)
                        
                        ax2.tick_params(axis = 'x', direction = 'in', length = 2.0)
                        plt.figtext(0.4, -0.028,'Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 10)
                        plt.figtext(-0.05, .77, 'Relative Normalized Equivalent Width (km)', rotation = 'vertical', fontsize = 10)
                        print 'FINISH FIGURE'
                        plt.savefig(os.path.join(ringutil.ROOT, 'chain_images', 'chain_' + num_id + '.png'), bbox_inches='tight', dpi = 200)

                           
                num +=1




c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

plot_clumps(c_approved_list, c_approved_db)

