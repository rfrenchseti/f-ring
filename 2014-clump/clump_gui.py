'''
Created on Jul 17, 2012

@author: rfrench
'''

from optparse import OptionParser
import pickle
import os.path
import ringutil
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from clumpmapdisp import ClumpMapDisp 
from Tkinter import *
import tkSimpleDialog
import clumputil
from imgdisp import ImageDisp, FloatEntry
import Image
import subprocess
from clump_radial_reproject import run_clump_radial_reproject
from imgdisp import ScrolledList
import numpy as np

#===============================================================================
#
# Command line processing
# 
#===============================================================================

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['-a']
#    cmd_line = ['--voyager']
#    cmd_line = ['--downsample']             

parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

#===============================================================================
# 
#===============================================================================

color_background = (0,0,0)
color_foreground = (1,1,1)
color_dark_grey = (0.5, 0.5, 0.5)
color_grey = (0.625, 0.625, 0.625)
color_bright_grey = (0.75, 0.75, 0.75)

figure_size = (10,7.5)
matplotlib.rc('figure', facecolor=color_background)
matplotlib.rc('axes', facecolor=color_background, edgecolor=color_foreground, labelcolor=color_foreground)
matplotlib.rc('xtick', color=color_foreground, labelsize=20)
matplotlib.rc('xtick.major', size=15)
matplotlib.rc('xtick.minor', size=9)
matplotlib.rc('ytick', color=color_foreground, labelsize=20)
matplotlib.rc('ytick.major', size=15)
matplotlib.rc('ytick.minor', size=9)
matplotlib.rc('font', size=20)
matplotlib.rc('legend', fontsize=10)
matplotlib.rc('text', color=color_foreground)
markersize = 18
markerwidth = 2

def fix_graph_colors(fig, ax, legend=None):
    for line in ax.xaxis.get_ticklines() + ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines() + ax.yaxis.get_ticklines(minor=True):
        line.set_color(color_foreground)
    if legend != None:
        legend.get_frame().set_facecolor(color_background)
        legend.get_frame().set_edgecolor(color_background)
        for text in legend.get_texts():
            text.set_color(color_foreground) 

#===============================================================================
# 
#===============================================================================

python_filename = sys.argv[0]
python_dir = os.path.split(sys.argv[0])[0]
python_radial_reproject_program = os.path.join(python_dir, ringutil.PYTHON_RING_RADIAL_REPROJECT)
# Compute longitude step of EW data
long_res = options.longitude_resolution*options.mosaic_reduction_factor

class MsgDialog(tkSimpleDialog.Dialog):
    def body(self, master):

        self.msg = Message(master, text="Text").grid(row=0)
        return

    def apply(self):
        pass

#===============================================================================
# 
# COMMAND BUTTONS THAT DEAL WITH CLUMPS
#
#===============================================================================

# Load the clump DB from a file
#
def command_load_clumps_from_file(clump_disp_data):
    clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
    clump_database_fp = open(clump_database_path, 'rb')
    clump_find_options = pickle.load(clump_database_fp)
    clump_disp_data.clump_db = pickle.load(clump_database_fp)
    clump_database_fp.close()
    print '** Clump Database created with: Scale min', clump_find_options.scale_min, 'Scale max', clump_find_options.scale_max,
    print 'Scale step', clump_find_options.scale_step
    print '*** Clumpsize min', clump_find_options.clump_size_min, 'Clumpsize max', clump_find_options.clump_size_max,
    print 'Prefilter', clump_find_options.prefilter
    print

    clump_disp_data.cmd.update_clump_db(clump_disp_data.clump_db)

def command_plot_displayed_clumps(clump_disp_data):
    clumputil.plot_ew_profiles_in_range(clump_disp_data.clump_db,
                                        clump_disp_data.cmd.disp_long_min,
                                        clump_disp_data.cmd.disp_long_max,
                                        clump_disp_data.cmd.disp_et_min,
                                        clump_disp_data.cmd.disp_et_max,
                                        ew_color=color_foreground)

    
#===============================================================================
# 
# COMMAND BUTTONS THAT DEAL WITH CLUMP CHAINS
#
#===============================================================================

#
# Load the clump chain DB from a file
#
def command_load_chains_from_file(clump_disp_data):
    clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
    clump_chains_fp = open(clump_chains_path, 'rb')
    clump_find_options2 = pickle.load(clump_chains_fp)
    clump_chain_options = pickle.load(clump_chains_fp)
    clump_disp_data.clump_db, clump_disp_data.clump_chain_list = pickle.load(clump_chains_fp)
    clump_chains_fp.close()
    print '** Clump Chain Database created with find options: Scale min', clump_find_options2.scale_min, 'Scale max', clump_find_options2.scale_max,
    print 'Scale step', clump_find_options2.scale_step
    print '*** Clumpsize min', clump_find_options2.clump_size_min, 'Clumpsize max', clump_find_options2.clump_size_max,
    print 'Prefilter', clump_find_options2.prefilter
    
    print '*** Clumps tracked with: Clumpsize min', clump_chain_options.clump_size_min,
    print 'Clumpsize max', clump_chain_options.clump_size_max,
    print 'Longitude tolerance', clump_chain_options.longitude_tolerance
    print '*** Max movement', clump_chain_options.max_movement*86400, 'Max time', clump_chain_options.max_time
    print
 
    clump_disp_data.cmd.update_clump_db(clump_disp_data.clump_db, redraw=False) # Only redraw once
    clump_disp_data.cmd.update_chain_list(clump_disp_data.clump_chain_list)


def command_load_approved_chain_list(clump_disp_data):

    if clump_disp_data.voyager_option:
        filename = 'voyager_approved_clumps_list.pickle'
    elif clump_disp_data.downsample_option:
        filename = 'downsample_approved_clumps_list.pickle'
    elif (clump_disp_data.downsample_option == False) or (clump_disp_data.voyager_option == False):
        filename = 'matrix_clumps_list.pickle'
        
    approved_file = os.path.join(ringutil.ROOT, 'clump-data',filename )

    if os.path.exists(approved_file):
        
        approved_file_fp = open(approved_file, 'rb')
        clump_db, approved_list = pickle.load(approved_file_fp)
        approved_file_fp.close()
        print len(approved_list)
        
    else:
#        approved_file_fp = open(approved_file, 'wb')
        approved_list = []

    clump_disp_data.approved_chain_list = approved_list
    print 'LOADING APPROVED CHAIN LIST', filename, len(approved_list)
    
    
def command_save_approved_chain_list(clump_disp_data):

    if clump_disp_data.voyager_option:
        filename = 'voyager_approved_clumps_list.pickle'
    elif clump_disp_data.downsample_option:
        filename = 'downsample_approved_clumps_list.pickle'
    elif (clump_disp_data.downsample_option == False) or (clump_disp_data.voyager_option == False):
        filename = 'matrix_clumps_list.pickle'
        
    approved_file = os.path.join(ringutil.ROOT, 'clump-data',filename )
    approved_file_fp = open(approved_file, 'wb')
    #save new chain list
    pickle.dump((clump_disp_data.clump_db, clump_disp_data.approved_chain_list), approved_file_fp)
    clump_disp_data.approved_chain_list = []
    approved_file_fp.close()
    print 'SAVING APPROVED CLUMP CHAIN LIST'
    
def command_delete_last_approved_chain(clump_disp_data):
    
    clump_disp_data.approved_chain_list = clump_disp_data.approved_chain_list[:-1]
#
# Load the clump chain DB from a file
#
def command_remove_all_chains(clump_disp_data):
    clump_disp_data.clump_chain_list = []
    clump_disp_data.cmd.update_chain_list(clump_disp_data.clump_chain_list)
    
#
# Create a new clump chain from the selected clumps and add to the current DB       
#
def command_create_chain_from_selected(clump_disp_data):
    sel_clumps = clump_disp_data.cmd.selected_clumps()
    # First sort it by ET
    sel_clumps.sort(key=lambda x: x.clump_db_entry.et)
    # XXX Error check for duplicate ETs
    new_chain = clumputil.ClumpChainData()
    new_chain.clump_list = sel_clumps
    new_chain.selected = True
    new_chain.rate, new_chain.base_long, res_fit = clumputil.fit_rate(new_chain.clump_list)
    new_chain.long_err_list = res_fit
    clump_disp_data.clump_chain_list.append(new_chain)
    clump_disp_data.cmd.update_chain_list(clump_disp_data.clump_chain_list)

#
# Walk the clumps in the displayed region to build new chains based on the current parameters
#
def command_track_displayed_clumps(clump_disp_data):
    # First mark everything as ignored
    for obsid, clump_db_entry in clump_disp_data.clump_db.items():
        for clump in clump_db_entry.clump_list:
            clump.ignore_for_chain = True
    # Second get the list of clumps currently displayed and mark them as not ignored
    for clump in clump_disp_data.cmd.displayed_clumps():
        if clump_disp_data.var_chain_clumpsize_min.get() <= clump.scale <= clump_disp_data.var_chain_clumpsize_max.get():
            clump.ignore_for_chain = False

    max_movement = 1.
    longitude_tolerance = 1.
    max_time = 180.
    
    new_clump_chain_list = clumputil.track_clumps(clump_disp_data.clump_db, 
                                                  clump_disp_data.var_chain_max_rate.get()/86400.,
                                                  clump_disp_data.var_chain_longitude_tolerance.get(),
                                                  clump_disp_data.var_chain_max_time.get()*86400., clump_disp_data.var_chain_scale_tolerance.get())

    clump_disp_data.clump_chain_list += new_clump_chain_list
    clump_disp_data.cmd.update_chain_list(clump_disp_data.clump_chain_list)
    
        
#
# Learn everything we can about a particular clump chain
#
def clump_chain_info(chain):
    # First sort it by ET
    chain.clump_list.sort(key=lambda x: x.clump_db_entry.et)
    
    retstr = ''
    retstr += 'Number of clumps: %d' % len(chain.clump_list) +'\n'
    retstr += 'Straight-line rate: %e deg/sec = %.8f deg/day ' % (chain.rate, chain.rate*86400.)
    retstr += '(a=%.4f km)' % ringutil.RelativeRateToSemimajorAxis(chain.rate) + '\n'
    retstr += 'Straight-line starting long: %.2f' % chain.base_long + '\n'
    
    for num, clump in enumerate(chain.clump_list):
        retstr += 'Clump %2d Long %6.2f (%6.2f) Width %6.2f ' % (num, clump.g_center, 
                                                                 chain.long_err_list[num],
                                                                 clump.fit_width_deg)
        retstr += '\n'
        
    return retstr

def command_display_chain_info(clump_disp_data):
    sel_chains = clump_disp_data.cmd.selected_chains()
    allstr = ''
    for chain in sel_chains:
        retstr = clump_chain_info(chain)
        allstr += retstr
    print allstr

#displays clumps in a selected chain    
def command_display_chain_profiles(clump_disp_data):
#    long_min = clump_disp_data.cmd.disp_long_min
#    long_max = clump_disp_data.cmd.disp_long_max
    long_min = 0.0
    long_max = 360.
    try:
        chain = clump_disp_data.sel_chain
        sub_size = len(chain.clump_list)
    except:
        print 'Did you update the chain list box? Please Select a Chain from There'
        return
    else:
        allstr = ''
        
#        sub_size = len(chain.clump_list)
        fig, axes = plt.subplots(sub_size, sharex=False)
        if sub_size == 1:
            axes = [axes] # Don't know why they return a single axis in this case
    
        axes = axes[::-1] # Earliest on the bottom
        for i, clump in enumerate(chain.clump_list):
            obsid = clump.clump_db_entry.obsid
            ew_data = clump.clump_db_entry.ew_data
            clumputil.plot_single_ew_profile(axes[i], clump.clump_db_entry, long_min, long_max, label=True, color=color_foreground)
            clumputil.plot_single_clump(axes[i], ew_data, clump, long_min, long_max, color= 'red')
            clumputil.plot_fitted_clump_on_ew(axes[i], ew_data, clump)
            axes[i].legend([obsid], loc=1, frameon=False)
            axes[i].set_xlim(long_min, long_max)
            
        retstr = clump_chain_info(chain)
        allstr
        print allstr
        plt.show()
    
def command_display_chain_rad_proj(clump_disp_data):
#    sel_chains = clump_disp_data.cmd.selected_chains()
    try:
        chain = clump_disp_data.sel_chain
    except:
        print 'Did you update the chain list box? Please Select a Chain from There'
        return
    else:
#        for chain in sel_chains:
        run_clump_radial_reproject(chain, options, Toplevel())

def command_add_ok_chain(clump_disp_data):
#    sel_chains = clump_disp_data.cmd.selected_chains()
    try:
        chain = clump_disp_data.sel_chain
    except:
        print 'Did you update the chain list box? Please Select a Chain from There'
        return
    else:
#        for chain in sel_chains:
        clump_disp_data.approved_chain_list.append(chain)
        print 'CHAIN HAS BEEN ADDED TO APPROVED LIST'

def command_update_chain_listbox(clump_disp_data):
    
    #clear everything first
    clump_disp_data.listbox_chains.clear()
    clump_disp_data.sel_chain = None

    sel_chains = clump_disp_data.cmd.selected_chains()
#    clump_disp_data.sel_chains_copy = sel_chains
    i = 0
    if sel_chains == None:
        return
    else:
        for chain in sel_chains:
            chain_num = 'chain_' + str(i) 
            clump_disp_data.listbox_chains.append(chain_num)
            i += 1

def command_compare_clump_inert_peri(clump_disp_data):
    sel_chains = clump_disp_data.cmd.selected_chains()
    
    if sel_chains == None:
        return

    else:
        for chain in sel_chains:
            i = 0
            while i <= (len(chain.clump_list)-2):
                
                clump1 = chain.clump_list[i]
                clump2 = chain.clump_list[i+1]
                ew_data1 = clump1.clump_db_entry.ew_data
                ew_data2 = clump2.clump_db_entry.ew_data
                longres1 = 360./len(ew_data1)
                longres2 = 360./len(ew_data2)
                (clump1_inert, clump1_per, clump1_true) = clumputil.compare_inert_to_peri(options, clump1.clump_db_entry.obsid, clump1.g_center/longres1, clump1.g_center)
                (clump2_inert, clump2_per, clump2_true) = clumputil.compare_inert_to_peri(options, clump2.clump_db_entry.obsid, clump2.g_center/longres2, clump2.g_center)
#                
#                print 'NUMBER OF ORBITS:', 
                print 'CLUMP 1 PERICENTRE:', clump1_per, 'ClUMP 1 INERTIAL LONG:', clump1_inert, 'CLUMP 1 TRUE:', clump1_true
                print 'CLUMP 2 PERICENTRE:', clump2_per, 'ClUMP 2 INERTIAL LONG:', clump2_inert, 'CLUMP 2 TRUE:', clump2_true
                
                i += 1
                print i

            
def get_selected_chain_from_list(chain_selection, clump_disp_data):
    
    #format of chain_selection: 'chain_#'
    chain_index = int(chain_selection[-1])
    chains = clump_disp_data.cmd.selected_chains()
    return chains[chain_index]
    

def offrep_obsid_list_buttonrelease_handler(event, clump_disp_data):
    chain_selections = clump_disp_data.listbox_chains.listbox.curselection()
    if len(chain_selections) == 0:
        return
    chain_selection = clump_disp_data.listbox_chains[int(chain_selections[0])][4:]
    clump_disp_data.sel_chain = get_selected_chain_from_list(chain_selection, clump_disp_data)
        
#===============================================================================
# 
#===============================================================================

class ClumpDispData(object):
    pass

clump_disp_data = ClumpDispData()

toplevel_frame = Frame(None)
left_control_frame = Frame(toplevel_frame)
right_disp_frame = Frame(toplevel_frame)

# Create all of the left-side controls
gridrow = 0
gridcol = 0

button_update = Button(left_control_frame, text='Load Clumps From File',
                       command=lambda: command_load_clumps_from_file(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Plot Displayed Clumps',
                       command=lambda: command_plot_displayed_clumps(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Load Chains From File',
                       command=lambda: command_load_chains_from_file(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Remove All Chains',
                       command=lambda: command_remove_all_chains(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Create Chain From Selected',
                       command=lambda: command_create_chain_from_selected(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Track Displayed Clumps',
                       command=lambda: command_track_displayed_clumps(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1


#------------------------------------------------------------------------------------------
#                        Chain Manipulation Functions/Actions
#------------------------------------------------------------------------------------------

#Create List Box for chain selection

button_update = Button(left_control_frame, text='Update Chain List',
                       command=lambda: command_update_chain_listbox(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

#Create List Box for chain selection

label = Label(left_control_frame, text='Selected Chains:')
label.grid(row=gridrow, column=0)
gridrow +=1

clump_disp_data.listbox_chains = ScrolledList(left_control_frame, width=20, height = 10,
                                     selectmode=BROWSE, font=('Courier', 10))
clump_disp_data.listbox_chains.listbox.bind("<ButtonRelease-1>",
        lambda event, clump_disp_data = clump_disp_data:
        offrep_obsid_list_buttonrelease_handler(event, clump_disp_data))
clump_disp_data.listbox_chains.grid(row=gridrow, column=0)
gridrow +=1

button_update = Button(left_control_frame, text='Display Chain Info',
                       command=lambda: command_display_chain_info(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Display Chain Profiles',
                       command=lambda: command_display_chain_profiles(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Plot Rad Proj of Selected Chain',
                       command=lambda: command_display_chain_rad_proj(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 2

button_update = Button(left_control_frame, text='Compare Inertial and Pericentre Longitudes',
                       command=lambda: command_compare_clump_inert_peri(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 2

#-------------------------------------------------------------------------------------------

clump_disp_data.var_chain_clumpsize_min = DoubleVar()
clump_disp_data.var_chain_clumpsize_min.set(0.1)
clump_disp_data.var_chain_clumpsize_max = DoubleVar()
clump_disp_data.var_chain_clumpsize_max.set(30.)
clump_disp_data.var_chain_max_time = DoubleVar()
clump_disp_data.var_chain_max_time.set(365.)
clump_disp_data.var_chain_max_rate = DoubleVar()
clump_disp_data.var_chain_max_rate.set(1.)
clump_disp_data.var_chain_longitude_tolerance = DoubleVar()
clump_disp_data.var_chain_longitude_tolerance.set(1.)
clump_disp_data.var_chain_scale_tolerance = DoubleVar()
clump_disp_data.var_chain_scale_tolerance.set(5.)

label = Label(left_control_frame, text='Chain Clump Size Min:')
label.grid(row=gridrow, column=0, sticky=W)
gridrow += 1
scale = Scale(left_control_frame, orient=HORIZONTAL, resolution=0.1,
              from_=0.1, to=50.,
              variable=clump_disp_data.var_chain_clumpsize_min)
scale.grid(row=gridrow, column=0, sticky=W)
gridrow += 1

label = Label(left_control_frame, text='Chain Clump Size Max:')
label.grid(row=gridrow, column=0, sticky=W)
gridrow += 1
scale = Scale(left_control_frame, orient=HORIZONTAL, resolution=0.1,
              from_=0.1, to=50.,
              variable=clump_disp_data.var_chain_clumpsize_max)
scale.grid(row=gridrow, column=0, sticky=W)
gridrow += 1

label = Label(left_control_frame, text='Chain Max Time (days):')
label.grid(row=gridrow, column=0, sticky=W)
gridrow += 1
scale = Scale(left_control_frame, orient=HORIZONTAL, resolution=1.,
              from_=1., to=365.,
              variable=clump_disp_data.var_chain_max_time)
scale.grid(row=gridrow, column=0, sticky=W)
gridrow += 1

label = Label(left_control_frame, text='Chain Max Rate (deg/day):')
label.grid(row=gridrow, column=0, sticky=W)
gridrow += 1
scale = Scale(left_control_frame, orient=HORIZONTAL, resolution=0.01,
              from_=0.01, to=1.0,
              variable=clump_disp_data.var_chain_max_rate)
scale.grid(row=gridrow, column=0, sticky=W)
gridrow += 1

label = Label(left_control_frame, text='Chain Longitude Tolerance:')
label.grid(row=gridrow, column=0, sticky=W)
gridrow += 1
scale = Scale(left_control_frame, orient=HORIZONTAL, resolution=0.1,
              from_=0.1, to=5.0,
              variable=clump_disp_data.var_chain_longitude_tolerance)
scale.grid(row=gridrow, column=0, sticky=W)
gridrow += 1

label = Label(left_control_frame, text='Chain Scale Tolerance:')
label.grid(row=gridrow, column=0, sticky=W)
gridrow += 1
scale = Scale(left_control_frame, orient=HORIZONTAL, resolution=0.1,
              from_=1.0, to=5.0,
              variable=clump_disp_data.var_chain_scale_tolerance)
scale.grid(row=gridrow, column=0, sticky=W)
gridrow += 2


#I want these AWAY from the other buttons so they don't get hit accidentally

button_update = Button(left_control_frame, text='Load Approved Chain List',
                       command=lambda: command_load_approved_chain_list(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Approve Selected Chain',
                       command=lambda: command_add_ok_chain(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Save Approved Chain List',
                       command=lambda: command_save_approved_chain_list(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1

button_update = Button(left_control_frame, text='Delete Last Chain',
                       command=lambda: command_delete_last_approved_chain(clump_disp_data))
button_update.grid(row=gridrow, column=0)
gridrow += 1



# Create the main clump display
clump_disp_data.clump_chain_list = []
clump_disp_data.clump_db = {}

clump_disp_data.cmd = ClumpMapDisp(clump_disp_data.clump_db, chain_list=clump_disp_data.clump_chain_list,
                                   parent=right_disp_frame)

#command_load_chains_from_file(clump_disp_data)

if options.voyager:
    clump_disp_data.voyager_option = True
else:
    clump_disp_data.voyager_option = False
    
if options.downsample:
    clump_disp_data.downsample_option = True
else:
    clump_disp_data.downsample_option = False


#clump_disp_data.approved_chain_list = command_load_approved_chain_list(clump_disp_data)

left_control_frame.grid(row=0, column=0, sticky=NW)
right_disp_frame.grid(row=0, column=1, sticky=NW)

toplevel_frame.pack()

mainloop()
