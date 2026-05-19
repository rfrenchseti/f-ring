import sys
import os
import getopt
import string
import ringutil
import pickle
import numpy as np
import numpy.ma as ma
from optparse import OptionParser

class MosaicData:
    def __init__(self):
        self.obsid = None
        self.obsid_list = None
        self.image_name_list = None
        self.image_path_list = None
        self.repro_path_list = None
        self.img = None
        self.longitudes = None
        self.radius = None
        self.resolutions = None
        self.image_numbers = None
        self.ETs = None
        self.emission_angles = None
        self.incidence_angles = None
        self.phase_angles = None

cmd_line = sys.argv[1:]
if len(cmd_line) == 0:
    
#    cmd_line = ['ISS_059RF_FMOVIE002_VIMS', 
#                '--opath', '/home/shicks/Documents/Movie_png'
#                ]
    cmd_line = ['-a', 
                '--ipath', '/home/shannon/test_Movie/',
                '--opath','/home/shannon/test_Movie/',
                '--fps', '1',
                '--out', 'Clump_Movement'
                ]
#    cmd_line = ['-a']

parser = OptionParser()
ringutil.add_parser_options(parser)

parser.add_option('--out', dest = 'output_file', type = 'string', default = 'animation')
parser.add_option('--ipath', dest = 'input_path', type = 'string', default = './')
parser.add_option('--opath', dest = 'output_path', type = 'string', default = './')
parser.add_option('--format', dest = 'format', type = 'string', default = '.png')
parser.add_option('--fps', dest = 'fps', type = 'int', default = 2)
parser.add_option('--usage', dest = 'usage', action = 'store_true', default = False)

options, args = parser.parse_args(cmd_line)
mosaicdata = MosaicData()


#####################################################################
#  Define a usage program
#####################################################################
def usage():

    print "\nPython script to create 2 animations: an mp4 and an avi.\n"
    print "Usage:\n"
    print "    ./animation.py [options]\n"
    print "    more descriptions if necessary...\n"
    print "    -o <output>            Set movie basename to 'output'"
    print "      --output=<output>      final name will be 'output'.avi"
    print "                             and 'output'.mp4\n"
    print "    -i <path>             Relative path to images"
    print "      --input-from=<path>\n"
    print "    -f <format>            Specify image format, options include:"
    print "      --img-format=<format>  png or jpg, others will be converted\n"
    print "    -x <fps>               Specify frames per second as 'fps'"
    print "      --fps=<fps>\n"
    print "    -h, --help             Display this help message\n"

def get_sorted_obsid_list(clump_db):
    obsid_by_time_db = {}    
    for obs_id in clump_db.keys():
        max_et = clump_db[obs_id].et_max
        obsid_by_time_db[max_et] = obs_id
    sorted_obsid_list = []
    for et in sorted(obsid_by_time_db.keys()):
        sorted_obsid_list.append(obsid_by_time_db[et])
    return sorted_obsid_list

def main_program(clump_db, output, output_path, input_path, img_format, fps):

    print "\nGenerating Animations:\n\n"

    print "    Please Wait...\n\n"

    if options.usage:
        usage()

    try: test=int(options.fps)
    except ValueError:
        print "\n---ERROR: Frames per second must be integer---\n"
        usage()
        sys.exit(2)
    # make sure it ends with a "/"
    if (input_path != ""):
        if (input_path[-1] != "/"):
            input_path = input_path + "/"

    # find the images
    list_name = "list_images_.txt"
    if os.path.isfile(list_name):
       os.remove(list_name)

    #grab and reorder the list of files according to time
    image_list = []
    for (path,dirs, files) in os.walk(input_path):
#    print path, dirs, files    
        for file in files:
            ind = string.rfind(file, '.png')
            if ind > 0:
                image_list.append(file[0:-4])
    
    print image_list
  
    full_sorted_ids = get_sorted_obsid_list(clump_db)
#    print sorted(db_by_time.keys())
    sorted_ids = []
    
    for key in full_sorted_ids:
        if key in image_list:
#        obsid = db_by_time[key]
            print 'CHECK:', key
            sorted_ids.append(key)
    print sorted_ids
    
    write_file = open(list_name, 'w')
    for obsid in sorted_ids:
        print input_path, obsid
        line = input_path + obsid + '.png\n'
        write_file.write(line)
    write_file.close()

    # create movie output names
    output1 = output + ".avi"
    output2 = output + ".mp4"

    # remove old movies
    if os.path.isfile(output1):
       os.remove(output1)

    if os.path.isfile(output2):
       os.remove(output2)

    # convert jpg files to png
    if (img_format == "jpg"):

        jpg_imgs = []

        f=open(list_name,'r')

        for line in f:
           jpg_imgs.append(line)

        f.close()

        print "Converting jpgs into pngs:\n"
        i=1
        for img in jpg_imgs:
            os.system("convert " + img[0:-1] + " " + img[0:-4] + "png")
            if (i % 25 == 0):
                print "   Image "+str(i)+" of "+str(len(jpg_imgs)+1)
            i += 1

        os.system("ls -v " + input_path + "*.png > " + list_name)

    # begin the movie making for png
    string1 = "mencoder mf://@" + list_name + " -ovc lavc -lavcopts "
    string2 = "vcodec=msmpeg4v2:vbitrate=3000:vhq -mf type=png:"
    string3 = "fps=" + str(fps) + " -o " + output_path + output1

    # generate avi:
    os.system(string1+string2+string3)

    # ONLY FOR THE MP4:
    # for some reason, the H.264 format skips the first ~28 files, so
    # duplicate the first 28 images in temp_.txt
    # This was not a problem when I was testing it, but I have seen it.
    # If it turns out to be a problem uncomment the next 4 lines and 
    # switch the list_name in string1 to images_h.264.txt
#    os.system("sed -n '1,28p' "+list_name+" > temp_.txt")
#    os.system("if [ $? -ne 0 ]; then echo 'Failed to copy 28 images'; fi")
#    os.system("cat temp_.txt "+list_name+" > images_h.264.txt")
#    os.system("if [ $? -eq 0 ]; then rm -rf temp_.txt; fi")

    # generate mp4:
    string1 = "mencoder mf://@"+list_name+" -of lavf -lavfopts "
    #string1 = "mencoder mf://@images_h.264.txt -of lavf -lavfopts "
    string2 = "format=mp4 -ss 1 -ovc x264 -x264encopts crf=20.0:"
    string3 = "nocabac:level_idc=30:global_header:threads=2 "
    string4 = "-fps "+str(fps)+" -o "+output_path + output2

    os.system(string1+string2+string3+string4)

    # if successful, remove the list_name
   # os.system("if [ $? -eq 0 ]; then rm -rf images_h.264.txt "
    #              +list_name+"; fi")


    print "\n\nThe output movies are:\n"
    print     "        Movie 1: " + output1
    print     "        Movie 2: " + output2

    print "\n\n---Complete---\n"

'''
-------------------------------------------------------
        RUN PROGRAM
-------------------------------------------------------
'''
clump_db_path, clump_chains_path = ringutil.clumpdb_paths(options)
clump_db_fp = open(clump_db_path, 'rb')
clump_find_options = pickle.load(clump_db_fp)
clump_db = pickle.load(clump_db_fp)
clump_db_fp.close()

    # call main program with appropriate arguments
main_program(clump_db, options.output_file, options.output_path, options.input_path, options.format, options.fps)
    
