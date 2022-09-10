import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import f_ring_util

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []
    # cmd_line = ['--voyager']

parser = argparse.ArgumentParser()

parser.add_argument('--voyager', action='store_true', default=False,
                    help='Process Voyager profiles instead of Cassini')
parser.add_argument('--ew-inner-radius', type=int, default=None,
                    help="""The inner radius of the range""")
parser.add_argument('--ew-outer-radius', type=int, default=None,
                    help="""The outer radius of the range""")

f_ring_util.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

def max_range(a, frac=0.9):
    target = np.sum(a)*frac
    all_ones = np.ones(len(a), dtype=float)
    for width in range(1, len(a)+1):
        conv = np.convolve(a, all_ones[:width], mode='valid')
        conv[conv < target] = np.inf
        min_idx = np.argmin(conv)
        if conv[min_idx] != np.inf:
            return (min_idx, min_idx+width-1)
    return None

def max_range_simple(a):
    target = np.sum(a) * 0.9
    best_sum = 0
    best_range = None
    for width in range(len(a), 0, -1):
        for idx in range(0, len(a)-width+1):
            s = np.sum(a[idx:idx+width])
            if s <= target and s >= best_sum:
                best_sum = s
                best_range = (idx,idx+width)
        if best_range is not None:
            break
    return best_range

if arguments.voyager:
    # Phase Angle, Emission Angle, Incidence Angle, Days from Encounter*Secs/Day, Encounter Date
    # Closest encounter dates courtesy of: http://nssdc.gsfc.nasa.gov/nmc/SpacecraftQuery.jsp
    # Emission angles are weighted from Table 1 of Showalter 2004
    # Phase Angles are from Figure of Showalter 2004
    # Incidence angles
    voyager_data_dict = {}
    voyager_data_dict['V1I'] = ( 15.0,  81.36, 86.1, -2.0*86400, '1980-11-12T23:46:30')
    voyager_data_dict['V1O'] = (125.0,  64.5,  86.1,  2.5*86400, '1980-11-12T23:46:30')
    voyager_data_dict['V2I'] = ( 10.5,  79.0,  82.1, -4.7*86400, '1981-08-26T03:24:57')
    voyager_data_dict['V2O'] = ( 98.0, 109.71, 82.1,  4.5*86400, '1981-08-26T03:24:57')

    for root, dirs, files in os.walk(f_ring_util.VOYAGER_PATH):
        for file in files:
            if '.STACK' in file:
                filepath = os.path.join(root, file)
                f = open(filepath)
                longitudes = []
                EWs = []
                trash = []
                filename = file[:-6]

                for line in f:
                    vars = line.split()
                    long, ew, trash = vars
                    longitudes.append(float(long))
                    EWs.append(float(ew))

                longitudes = np.array(longitudes)
                EWs = np.array(EWs)/1000

                #fake data (Except emission and phase angles)
                et = cspyce.utc2et(voyager_data_dict[filename][4]) + voyager_data_dict[filename][3]
                print(cspyce.et2utc(et, 'C', 0))
                ETs = np.zeros(len(EWs)) + et
                resolutions = np.zeros(len(EWs)) - 9999
                image_numbers = np.zeros(len(EWs)) - 9999
                emission_angles = np.zeros(len(EWs)) + voyager_data_dict[filename][1]
                phase_angles = np.zeros(len(EWs)) + voyager_data_dict[filename][0]
                incidence_angles = np.zeros(len(EWs)) + voyager_data_dict[filename][2]

                #make metadata for voyager
                data_path, metadata_path, large_png_path, small_png_path = f_ring_util.mosaic_paths(arguments, filename)
                metadata = (longitudes, resolutions, image_numbers,
                            ETs, emission_angles, incidence_angles,
                            phase_angles)
                metadata_fp = open(metadata_path, 'wb')
                pickle.dump(metadata, metadata_fp)
                metadata_fp.close()

                #fake bkgnd data for voyager
                (reduced_mosaic_data_filename, bkgnd_sub_mosaic_filename,
                     bkgnd_mask_filename, bkgnd_model_filename,
                     bkgnd_sub_mosaic_metadata_filename) = f_ring_util.bkgnd_paths(arguments, filename)

                np.save(bkgnd_model_filename, [])     #just need the blank files
                np.save(bkgnd_mask_filename, [])
                bkgnd_data = (None, None, None, None, None, None, None, None, None)
                bkgnd_metadata_fp = open(bkgnd_sub_mosaic_metadata_filename, 'wb')
                pickle.dump(bkgnd_data, bkgnd_metadata_fp)
                bkgnd_metadata_fp.close()

                reduced_mosaic_metadata_fp = open(bkgnd_sub_mosaic_filename, 'wb')
                pickle.dump(metadata, reduced_mosaic_metadata_fp)
                pickle.dump([filename], reduced_mosaic_metadata_fp)
                pickle.dump([None], reduced_mosaic_metadata_fp)
                pickle.dump([None], reduced_mosaic_metadata_fp)
                reduced_mosaic_metadata_fp.close()

                fake_reduced_mosaic = np.zeros((10,10))+999
                reduced_mosaic_data_fp = open(reduced_mosaic_data_filename, 'wb')
                np.save(reduced_mosaic_data_fp, fake_reduced_mosaic)
                reduced_mosaic_data_fp.close()

                EWs /= np.abs(np.cos(emission_angles[0]*np.pi/180.)) # Convert normal to raw I/F
                EWs = EWs.view(ma.MaskedArray)
                EWs.mask = False
                EWs[np.where(EWs == 0.)] = ma.masked

                (ew_data_filename, ew_mask_filename) = f_ring_util.ew_paths(arguments, filename)

                np.save(ew_data_filename, EWs.data)
                np.save(ew_mask_filename, ma.getmask(EWs))
else:
    for obs_id in f_ring_util.enumerate_obsids(arguments):
        (bkgnd_sub_mosaic_filename,
         bkgnd_sub_mosaic_metadata_filename) = f_ring_util.bkgnd_sub_mosaic_paths(
            arguments, obs_id)
        (ew_data_filename, ew_metadata_filename) = f_ring_util.ew_paths(
            arguments, obs_id, make_dirs=True)

        if (not os.path.exists(bkgnd_sub_mosaic_filename) or
            not os.path.exists(bkgnd_sub_mosaic_metadata_filename)):
            print('NO FILE', bkgnd_sub_mosaic_filename,
                  'OR', bkgnd_sub_mosaic_metadata_filename)
            continue

        with open(bkgnd_sub_mosaic_metadata_filename, 'rb') as bkgnd_metadata_fp:
            metadata = pickle.load(bkgnd_metadata_fp, encoding='latin1')
        with np.load(bkgnd_sub_mosaic_filename) as npz:
            bkgnd_sub_mosaic_img = ma.MaskedArray(**npz)

        if arguments.ew_inner_radius is not None:
            ring_lower_limit = int((arguments.ew_inner_radius -
                                    arguments.radius_inner_delta -
                                    arguments.ring_radius) / arguments.radius_resolution)
        else:
            ring_lower_limit = 0
        if arguments.ew_outer_radius is not None:
            ring_upper_limit = int((arguments.ew_outer_radius -
                                    arguments.radius_inner_delta -
                                    arguments.ring_radius) / arguments.radius_resolution)
        else:
            ring_upper_limit = bkgnd_sub_mosaic_img.shape[0]-1
        # print(ring_upper_limit-ring_lower_limit)
        # assert False
        longitudes = metadata['longitudes']
        resolutions = metadata['resolutions']
        image_numbers = metadata['image_numbers']
        ETs = metadata['ETs']
        emission_angles = metadata['emission_angles']
        incidence_angle = metadata['incidence_angle']
        phase_angles = metadata['phase_angles']

        # longitudes = longitudes[::180]
        # bkgnd_sub_mosaic_img = bkgnd_sub_mosaic_img[:,::180]

        bad_long = longitudes < 0
        bad_long = bad_long | (np.sum(ma.getmaskarray(bkgnd_sub_mosaic_img), axis=0) >
                               bkgnd_sub_mosaic_img.shape[0]*.2)

        widths = np.zeros((len(longitudes), 2), dtype=float)
        for idx in range(len(longitudes)):
            if bad_long[idx] < 0:
                continue
            ret = max_range(bkgnd_sub_mosaic_img[ring_lower_limit:ring_upper_limit+1,
                                                 idx])
            widths[idx] = (ret[0]+ring_lower_limit, ret[1]+ring_lower_limit)

        width_data_filename = f_ring_util.width_paths(arguments, obs_id)
        f_ring_util.write_width(width_data_filename, widths)

        w = (widths[~bad_long,1]-widths[~bad_long,0])*arguments.radius_resolution
        print('%-30s Width %8.5f +/- %8.5f' % (
              obs_id, ma.mean(w), np.std(w)))

        if False:
            longitudes = longitudes.view(ma.MaskedArray)
            w = w.view(ma.MaskedArray)
            w[bad_long] = ma.masked
            longitudes[bad_long] = ma.masked
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.degrees(longitudes), widths[:,0])
            plt.plot(np.degrees(longitudes), widths[:,1])
            plt.xlim(0,360)
            plt.show()
