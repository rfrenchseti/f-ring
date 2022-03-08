################################################################################
# main_clump_track.py
#
# Purpose: Locate and store clumps that move over time.
# Clumps that move over time will be stored in a dictionary.
# Info needed: Obsid's it shows up in. Center Longitudes over time.
################################################################################

import argparse
import pickle
import sys

import clump_util
import ringutil

cmd_line = sys.argv[1:]
if len(cmd_line) == 0:
    cmd_line = ['--clump-size-min', '3.5', '--clump-size-max', '100',
                '--longitude-tolerance', '.5',
                '--max-movement', '0.7',
                '--scale-tolerance', '2.0',
                '-a',
#                '--voyager'
#                '--downsample'

#                '--first-obsid', 'ISS_006RI_LPHRLFMOV001_PRIME',
#                '--last-obsid', 'ISS_006RI_LPHRLFMOV001_PRIME',
               ]

parser = argparse.ArgumentParser()

parser.add_argument('--clump-size-min', type=float, dest='clump_size_min', default=5.,
                    help='The minimum clump size (in full-width degrees) to detect')
parser.add_argument('--clump-size-max', type=float, dest='clump_size_max', default=15.,
                    help='The maximum clump size (in full-width degrees) to detect')
parser.add_argument('--first-obsid', dest='first_obsid', default='',
                    help='The first OBSID to use')
parser.add_argument('--last-obsid', dest='last_obsid', default='',
                    help='The last OBSID to use')
parser.add_argument('--longitude-tolerance', type=float, dest='longitude_tolerance', default=0.5,
                    help='How close a clump in a chain has to match in longitude')
parser.add_argument('--max-movement', type=float, dest='max_movement', default=0.3,
                    help='Maximum allowed movement in deg/day')
parser.add_argument('--max-time', type=float, dest='max_time', default=90.,
                    help='Maximum time duration of a chain in days')
parser.add_argument('--scale-tolerance', type=float, dest='scale_tolerance', default= 2.,
                    help='Maximum fractional change in width from clump-to-clump')

ringutil.add_parser_options(parser)
arguments = parser.parse_args(cmd_line)


clump_db_path, clump_chains_path = ringutil.clumpdb_paths(arguments)
with open(clump_db_path, 'rb') as clump_db_fp:
    clump_find_options = pickle.load(clump_db_fp)
    clump_db = pickle.load(clump_db_fp)

print('** Clump db created with: Scale min', clump_find_arguments.scale_min,
        'Scale max', clump_find_arguments.scale_max,
        'Scale step', clump_find_arguments.scale_step)
print('*** Clumpsize min', clump_find_arguments.clump_size_min,
        'Clumpsize max', clump_find_arguments.clump_size_max,
        'Prefilter', clump_find_arguments.prefilter)


#=======================================================================
#
# MAIN PROGRAM
#
#=======================================================================

et_first_obsid = 0
et_last_obsid = 1e38
if arguments.first_obsid != '':
    et_first_obsid = clump_db[arguments.first_obsid].et
if arguments.last_obsid != '':
    et_last_obsid = clump_db[arguments.last_obsid].et

# Now restrict the clumps to only those of the proper sizes
if not arguments.voyager:
    for obsid in clump_db.keys():
        for clump in clump_db[obsid].clump_list:
            clump.ignore_for_chain = (
                (not (arguments.clump_size_max >= clump.scale >= arguments.clump_size_min)) or
                 clump_db[obsid].et < et_first_obsid or
                 clump_db[obsid].et > et_last_obsid)

for obsid in clump_db.keys():
    clump_db_entry = clump_db[obsid]
    for clump in clump_db_entry.clump_list:
        clump.clump_db_entry = clump_db_entry

clump_chain_list = clumputil.track_clumps(clump_db, arguments.max_movement/86400.,
                                          arguments.longitude_tolerance,
                                          arguments.max_time*86400.,
                                          arguments.scale_tolerance)

clump_chain_options = clumputil.ClumpChainOptions()
clump_chain_arguments.clump_size_min = arguments.clump_size_min
clump_chain_arguments.clump_size_max = arguments.clump_size_max
clump_chain_arguments.longitude_tolerance = arguments.longitude_tolerance
clump_chain_arguments.max_time = arguments.max_time
clump_chain_arguments.max_movement = arguments.max_movement/86400.

with open(clump_chains_path, 'wb') as clump_chains_fp:
    pickle.dump(clump_find_options, clump_chains_fp)
    pickle.dump(clump_chain_options, clump_chains_fp)
    pickle.dump((clump_db, clump_chain_list), clump_chains_fp)
1
