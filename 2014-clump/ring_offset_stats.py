'''
Created on Oct 9, 2011

@author: rfrench
'''

from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import pickle
import ringutil
import sys
import os.path

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['-a']

parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

n_offset_x_list = []
n_offset_y_list = []
n_offset_file_list = []
w_offset_x_list = []
w_offset_y_list = []
w_offset_file_list = []

for obs_id, image_name, image_path in ringutil.enumerate_files(options, args, '_CALIB.IMG'):
    offset_path = ringutil.offset_path(options, image_path, image_name)
    if not os.path.exists(offset_path):
        continue

    offset_file_version, the_offset, manual_offset, fring_version = ringutil.read_offset(offset_path)

    correct_offset = the_offset
    if manual_offset != None:
        continue
#        correct_offset = manual_offset
    if correct_offset == None:
        continue

    if image_name[0] == 'N':
        n_offset_x_list.append(correct_offset[0])
        n_offset_y_list.append(correct_offset[1])
        n_offset_file_list.append(offset_path)
    else:
        w_offset_x_list.append(correct_offset[0])
        w_offset_y_list.append(correct_offset[1])
        w_offset_file_list.append(offset_path)

print 'NAC'
print 'X min', np.min(n_offset_x_list), n_offset_file_list[np.argmin(n_offset_x_list)]
print 'X max', np.max(n_offset_x_list), n_offset_file_list[np.argmax(n_offset_x_list)]
print 'Y min', np.min(n_offset_y_list), n_offset_file_list[np.argmin(n_offset_y_list)]
print 'Y max', np.max(n_offset_y_list), n_offset_file_list[np.argmax(n_offset_y_list)]
print 'WAC'
print 'X min', np.min(w_offset_x_list), w_offset_file_list[np.argmin(w_offset_x_list)]
print 'X max', np.max(w_offset_x_list), w_offset_file_list[np.argmax(w_offset_x_list)]
print 'Y min', np.min(w_offset_y_list), w_offset_file_list[np.argmin(w_offset_y_list)]
print 'Y max', np.max(w_offset_y_list), w_offset_file_list[np.argmax(w_offset_y_list)]
