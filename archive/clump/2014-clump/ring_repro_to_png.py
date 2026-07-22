'''
Created on Oct 10, 2011

@author: rfrench
'''

from optparse import OptionParser
import ringutil
import ringimage
import os
import os.path
import fitreproj
import vicar
import numpy as np
import sys
from imgdisp import ImageDisp, FloatEntry, ScrolledList
from PIL import Image

python_filename = sys.argv[0]

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['test_059RF']

parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

for obsid, image_name, image_path in ringutil.enumerate_files(options, args, '_CALIB.IMG'):
    repro_path = ringutil.repro_path(options, image_path, image_name)
    png_path = repro_path[:-4] + '.png'
    if (os.path.exists(repro_path) and
        (not os.path.exists(png_path) or os.stat(repro_path).st_mtime > os.stat(png_path).st_mtime)):
        print 'Processing', obsid, '/', image_name
        repro_vicar_data = vicar.VicarImage.FromFile(repro_path)
        repro_img = repro_vicar_data.Get2dArray()
        blackpoint = max(np.min(repro_img), 0)
        whitepoint = np.max(repro_img)
        gamma = 0.5
        # The +0 forces a copy - necessary for PIL
        int_image = np.cast['int8'](ImageDisp.ScaleImage(repro_img, blackpoint,
                                                         whitepoint, gamma))[::-1,:]+0
        img = Image.frombuffer('L', (int_image.shape[1], int_image.shape[0]),
                               int_image, 'raw', 'L', 0, 1)
        img.save(png_path, 'PNG')
