'''
Author: Shannon Hicks

A program to analyze the distribution of clump heights across profiles.
Question: How do we define clump heights?
1. Absolute Heights: the height at the center longitude times the mexhat height?
2. +x sigma away from the LOCAL mean?
'''
import pickle
import clumputil
import numpy as np
import numpy.ma as ma
import pickle
import sys
import os.path
import ringutil
#import cspice
import matplotlib.pyplot as plt
import matplotlib
from optparse import OptionParser
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['--whole-analysis', '--plot-fit']
#    cmd_line = ['--side-analysis', '--plot-fit']
#    cmd_line = ['--write-pdf']

parser = OptionParser()
ringutil.add_parser_options(parser)


parser.add_option('--whole-analysis', dest = 'whole_analysis',
                  action = 'store_true', default = False)
parser.add_option('--side-analysis', dest = 'side_analysis',
                  action = 'store_true', default = False)
parser.add_option('--plot-fit', dest = 'plot_fit',
                   action = 'store_true', default = False)
parser.add_option('--debug', dest = 'debug',
                  action = 'store_true', default = False)

parser.add_option('--write-pdf', dest='write_pdf',
                  action='store_true', default=False,
                  help='Write plots to a PDF file')

options, args = parser.parse_args(cmd_line)

if options.write_pdf:
    color_background = (1,1,1)
    color_foreground = (0,0,0)
    color_dark_grey = (0.5, 0.5, 0.5)
    color_grey = (0.375, 0.375, 0.375)
    color_bright_grey = (0.25, 0.25, 0.25)
    figure_size = (10.,7.5)
    matplotlib.rc('figure', facecolor=color_background)
    matplotlib.rc('axes', facecolor=color_background, edgecolor=color_foreground, labelcolor=color_foreground)
    matplotlib.rc('xtick', color=color_foreground, labelsize=10)
    matplotlib.rc('xtick.major', size=8)
    matplotlib.rc('xtick.minor', size=5)
    matplotlib.rc('ytick', color=color_foreground, labelsize=10)
    matplotlib.rc('ytick.major', size=8)
    matplotlib.rc('ytick.minor', size=5)
    matplotlib.rc('font', size=10)
    matplotlib.rc('legend', fontsize=5)
    matplotlib.rc('text', color=color_foreground)
    
def abs_height_distribution(clump_db, options):
      
    #need one large list of clump heights for the histogram. A 2D array shows up as multiple data sets
    clump_height_list = []
    for obsid in clump_db.keys():
        for clump in clump_db[obsid].clump_list:
            print 'MEXHAT', clump.mexhat_height, 'ABS', clump.abs_height
            
            clump_height_list.append(clump.abs_height)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    step = .025
    bins = np.arange(0, 10,step)
    
    counts, bins, patches = plt.hist(clump_height_list, bins)
    
    if options.debug:
        print counts, bins
  
    plt.title('Absolute Height Distribution')
    plt.xlabel('Clump Heights')
    
    plt.show()
    

def sigma_height_distribution(clump_db, options):
    
    clump_sigma_list = []
    longitudes = np.arange(0,360,.04)
    long_triple = np.tile(longitudes, 3)
    
    for obsid in clump_db.keys():
        ew_data = clump_db[obsid].ew_data
        ew_data_triple = np.tile(ew_data, 3)
        
        for clump in clump_db[obsid].clump_list:
            
            if options.side_analysis:
                outer_left = (clump.longitude_idx - clump.scale_idx/2.) + len(ew_data)
                outer_right = (clump.longitude_idx + clump.scale_idx/2.) + len(ew_data)
                
                left_range = ew_data_triple[outer_left - 2000:outer_left]
                right_range = ew_data_triple[outer_right:(outer_right + 2000)]
                outer_ew_range = left_range + right_range
                
                if options.debug:
                    print len(ew_data_triple[9000:len(ew_data_triple)-9000])
                    plt.plot(longitudes, ew_data_triple[9000:len(ew_data_triple)-9000], color = 'blue')
                    plt.plot(long_triple[outer_left-500:outer_left],left_range, color = 'r')
                    plt.plot(long_triple[outer_right:outer_right + 500],right_range, color = 'r')
                    plt.show()

                sigma = ma.std(outer_ew_range)
                subtitle = 'Side Analysis'
            
            if options.whole_analysis:
                sigma = ma.std(ew_data)
                subtitle = 'Whole Analysis'
                
            clump_height = ew_data[clump.longitude_idx]
#            clump_sigma = clump.abs_height/sigma
#            clump_sigma = clump.clump_sigma
#            print clump.fit_sigma
            clump_sigma = clump.fit_sigma
            clump_scale= clump.scale
#            print clump_sigma
            if options.debug:
                print 'PROFILE SIGMA', sigma,'HEIGHT', clump_height, 'CLUMP SIGMA', clump_sigma
            if clump_sigma > 1.0:
                clump_sigma_list.append(clump_scale)
    print len(clump_sigma_list)
    # CREATE FIGURE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    step = .025
    graph_max= ma.max(clump_sigma_list)
    graph_min = ma.min(clump_sigma_list)
    if options.debug:
        print 'MAX, MIN: ', graph_max, graph_min
        
    bins = np.arange(graph_min,graph_max,step)
    
    counts, bins, patches = plt.hist(clump_sigma_list, bins)
    if options.debug:
        print 'COUNTS, BINS: '
        print counts, bins
        
    if options.plot_fit:
        y = fit_curve(clump_sigma_list, bins)
        plt.plot(bins, y, 'w--', linewidth = 1.5, color = 'r')
        
    plt.title('Sigma Height Distribution: ' + 'CASSINI DATA')
    plt.xlabel('Sigma Clump Heights')
    
    plt.show()
    
    return clump_sigma_list

def clump_width_height_compare(clump_db, sigma_list, options):
    
    clump_widths = []
    clump_heights = []
    
    for obsid in clump_db.keys():
        for clump in clump_db[obsid].clump_list:
            width = clump.scale
            height = clump.abs_height
            
            clump_widths.append(width)
            clump_heights.append(height)
    
    print len(clump_widths), len(clump_heights)
    
    plt.plot(clump_widths, clump_heights, '.', color = 'r')
    plt.plot(clump_widths, sigma_list, '.', color = 'blue')
    
    plt.title('Width/Height Correlation')
    plt.xlabel('Clump Width (Degrees)')
    plt.ylabel('Clump Absolute Height, Clump Sigma Height')
    
    plt.show()
    
def fit_curve(data, bins):
    
    y = mlab.normpdf(bins, ma.mean(data), ma.std(data))
    
    return y

def show_or_save(pdf_file):
    if options.write_pdf:
        pdf_file.savefig(papertype='letter', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_height_pdf(clump_db, options):
    
    version ='_1'
    pdf1 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas1' + version +'.pdf')
    pdf2 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas2'+ version +'.pdf')
    pdf3 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas3'+ version +'.pdf')
    pdf4 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas4'+ version +'.pdf')
    pdf5 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas5'+ version +'.pdf')
    pdf6 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas6'+ version +'.pdf')
    pdf7 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas7'+ version +'.pdf')
    pdf8 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas8'+ version +'.pdf')
    pdf9 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas9'+ version +'.pdf')
    pdf10 = os.path.join(ringutil.ROOT, 'clumps_fit_sigmas10'+ version +'.pdf')
    
    
    sigma_dict = {}
    for obsid in clump_db.keys():
        for clump in clump_db[obsid].clump_list:
            sigma_dict[clump.fit_sigma] = [clump, obsid]
            
    #sort the clumps into separate lists based on height
    clumps1 = []
    clumps2 = []
    clumps3 = []
    clumps4 = []
    clumps5 = []
    clumps6 = []
    clumps7 = []
    clumps8 = []
    clumps9 = []
    clumps10 = []
    
    for clump_sigma in sorted(sigma_dict.keys()):
        if 0.0 < clump_sigma < 0.25:
            clumps1.append(sigma_dict[clump_sigma])
        if 0.25 < clump_sigma < 0.5:
            clumps2.append(sigma_dict[clump_sigma])
        if 0.5 < clump_sigma < 0.75:
            clumps3.append(sigma_dict[clump_sigma])
        if 0.75 < clump_sigma < 1.0:
            clumps4.append(sigma_dict[clump_sigma])
        if 1.0 < clump_sigma < 1.5:
            clumps5.append(sigma_dict[clump_sigma])
        if 1.5 < clump_sigma < 2.0:
            clumps6.append(sigma_dict[clump_sigma])
        if 2.0 < clump_sigma < 2.5:
            clumps7.append(sigma_dict[clump_sigma])
        if 4.0 < clump_sigma < 10.0:
            clumps8.append(sigma_dict[clump_sigma])
#        if 2.0 < clump_sigma < 3.0:
#            clumps_8_9.append(sigma_dict[clump_sigma])
#        if 3.0 < clump_sigma < 5.0:
#            clumps_9_20.append(sigma_dict[clump_sigma])
    #open file for clumps 0-1
    
    long_min = 0.
    long_max = 360.
    
    file_list = [pdf1, pdf2, pdf3, pdf4,
                  pdf5, pdf6, pdf7, pdf8]
    clump_lists = [clumps1, clumps2, clumps3, clumps4,
                    clumps5, clumps6, clumps7, clumps8]
#    file_list = [pdf_8_9]
#    clump_lists = [clumps_8_9]
    
    for i, pdf_filename in enumerate(file_list):
        print i
        pdf_file = PdfPages(pdf_filename)
        k = 0
        for pair in clump_lists[i]:
            clump, obsid = pair
            clump_db_entry = clump_db[obsid]
            
            if k == 4:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                clumputil.plot_single_ew_profile(ax, clump_db_entry, long_min, long_max)
                clumputil.plot_single_clump(ax, clump_db_entry.ew_data, clump, long_min, long_max) 
                ax.set_title(str(clump.fit_sigma) + ', ' + obsid)
                show_or_save(pdf_file)
                k = 0
            k += 1
        if options.write_pdf:
            pdf_file.close()
        
    
        

'''
---------------------- MAIN PROGRAM ------------------------------
'''

clump_db_path, clump_chains_path = ringutil.clumpdb_paths(options)
clump_db_fp = open(clump_db_path, 'rb')
clump_find_options = pickle.load(clump_db_fp)
clump_db = pickle.load(clump_db_fp)
clump_db_fp.close()


#plot_height_pdf(clump_db, options)
sigma_list = sigma_height_distribution(clump_db, options)
#abs_height_distribution(clump_db, options)
#clump_width_height_compare(clump_db, sigma_list, options)