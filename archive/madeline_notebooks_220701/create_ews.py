import argparse
import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.ndimage as nd

import f_ring_util.f_ring as f_ring
import julian

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []

parser = argparse.ArgumentParser()

parser.add_argument('--ew-inner-radius', type=int, default=139470,
                    help='The inner radius of the range')
parser.add_argument('--ew-outer-radius', type=int, default=140970,
                    help='The outer radius of the range')
parser.add_argument('--compute-widths', action='store_true', default=False,
                    help='Compute the widths for each slice')
parser.add_argument('--plot-results', action='store_true', default=False,
                    help='Plot the EW and Width results')
parser.add_argument('--save-plots', action='store_true', default=False,
                    help='Same as --plot-results but save plots to disk instead')
parser.add_argument('--downsample', type=int, default=1,
                    help='Amount to downsample the mosaic in longitude')
parser.add_argument('--minimum-coverage', type=float, default=60,
                    help='Minimum coverage (in degrees) allowed for a good obsid')
parser.add_argument('--maximum-bad-pixels-percentage', type=float, default=2,
                    help='Maximum percentage of bad pixels for a good radial slice')
parser.add_argument('--slice-size', type=int, default=0,
                    help='Slice size in degrees longitude')
parser.add_argument('--output-csv-filename', type=str,
                    help='Name of output CSV file')
parser.add_argument('--core-half-width', type=int, default=100,
                    help='The half-width of the core for computing ring width')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

wing1_width = int((arguments.ring_radius - arguments.core_half_width -
                   arguments.ew_inner_radius) /
                  arguments.radius_resolution)
wing2_width = int((arguments.ew_outer_radius - arguments.ring_radius -
                   arguments.core_half_width) /
                  arguments.radius_resolution)

##########################################################################################

def max_range(a, fracs, mean_brightness):
    ret = []
    frac_idx = 0
    if mean_brightness is None:
        mean_brightness = np.sum(a)
    target = mean_brightness*fracs[frac_idx]
    all_ones = np.ones(len(a), dtype=float)
    for width in range(1, len(a)+1):
        conv = np.convolve(a, all_ones[:width], mode='valid')
        conv[conv < target] = np.inf
        min_idx = np.argmin(conv)
        if conv[min_idx] != np.inf:
            ret.append((min_idx, min_idx+width-1))
            frac_idx += 1
            if frac_idx >= len(fracs):
                return ret
            target = mean_brightness*fracs[frac_idx]
    return None

def width_from_wings(a, fracs, mean_brightness=None):
    # a = nd.median_filter(a, 9)
    if mean_brightness is None:
        mask = np.ones(len(a), dtype=bool)
        mask[:wing1_width+1] = True
        mask[wing2_width:] = True
        mean_brightness = np.mean(a[mask])
    ret = []
    for frac in fracs:
        target = mean_brightness * frac
        wing1_pos = np.argmax(a >= target)
        wing2_pos = len(a) - np.argmax(a[::-1] >= target) - 1
        # print(wing1_pos, a[wing1_pos], wing2_pos, a[wing2_pos])
        ret.append((wing1_pos, wing2_pos))
    return ret

##########################################################################################

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
    ring_upper_limit = ((arguments.radius_outer_delta - arguments.radius_inner_delta) /
                        arguments.radius_resolution)

print(f'EW lower limit: {arguments.ew_inner_radius:d} km, pix {ring_lower_limit:4d}')
print(f'EW upper limit: {arguments.ew_outer_radius:d} km, pix {ring_upper_limit:4d}')

if arguments.output_csv_filename:
    assert arguments.slice_size == 0 or (360 % arguments.slice_size) == 0
    csv_fp = open(arguments.output_csv_filename, 'w')
    writer = csv.writer(csv_fp)
    hdr = ['Observation', 'Slice#', 'Num Data', 'Date', 'Min Long', 'Max Long',
           'Min Res', 'Max Res', 'Mean Res',
           'Min Phase', 'Max Phase', 'Mean Phase',
           'Min Emission', 'Max Emission', 'Mean Emission',
           'Incidence',
           '% Coverage',
           'EW', 'EW Std', 'Normal EW', 'Normal EW Std']
    if arguments.compute_widths:
        hdr += ['Width50', 'Width50 Std',
                'Width75', 'Width75 Std',
                'Width100', 'Width100 Std']
    writer.writerow(hdr)

for obs_id in f_ring.enumerate_obsids(arguments):
    if '166RI' in obs_id or '237RI' in obs_id:
        print(f'{obs_id:30s} SKIPPING')
        continue

    (bkgnd_sub_mosaic_filename,
     bkgnd_sub_mosaic_metadata_filename) = f_ring.bkgnd_sub_mosaic_paths(
        arguments, obs_id)

    if (not os.path.exists(bkgnd_sub_mosaic_filename) or
        not os.path.exists(bkgnd_sub_mosaic_metadata_filename)):
        print('NO FILE', bkgnd_sub_mosaic_filename,
              'OR', bkgnd_sub_mosaic_metadata_filename)
        continue

    with open(bkgnd_sub_mosaic_metadata_filename, 'rb') as bkgnd_metadata_fp:
        metadata = pickle.load(bkgnd_metadata_fp, encoding='latin1')
    with np.load(bkgnd_sub_mosaic_filename) as npz:
        bsm_img = ma.MaskedArray(**npz)

    ds = arguments.downsample

    longitudes = metadata['longitudes'][::ds].view(ma.MaskedArray)
    resolutions = metadata['resolutions'][::ds]
    image_numbers = metadata['image_numbers'][::ds]
    ETs = metadata['ETs'][::ds]
    emission_angles = metadata['emission_angles'][::ds]
    incidence_angle = metadata['incidence_angle']
    phase_angles = metadata['phase_angles'][::ds]

    restr_bsm_img = bsm_img[ring_lower_limit:ring_upper_limit+1,::ds]

    bad_long = longitudes < 0
    percentage_long_ok = float(np.sum(~bad_long)) / len(longitudes) * 100
    bad_long = bad_long | (np.sum(ma.getmaskarray(restr_bsm_img), axis=0) >
           restr_bsm_img.shape[0]*arguments.maximum_bad_pixels_percentage/100)
    percentage_ew_ok = float(np.sum(~bad_long)) / len(longitudes) * 100

    brightness = np.sum(restr_bsm_img, axis=0)
    mean_brightness = np.mean(brightness)
    ew_profile = brightness * arguments.radius_resolution
    ew_profile[bad_long] = ma.masked
    longitudes[bad_long] = ma.masked

    mask = np.zeros(restr_bsm_img.shape[0], dtype=bool)
    # mask = np.ones(restr_bsm_img.shape[0], dtype=bool)
    mask[:wing1_width+1] = True
    mask[-wing2_width:] = True
    # mask[wing1_width:wing2_width+1] = True
    # print(f'Full rad {len(mask)}, Wing1 {wing1_width}, Wing2 {wing2_width}')
    width_brightness = np.mean(restr_bsm_img[mask,:])

    ew_mean = ma.mean(ew_profile)
    ew_std = ma.std(ew_profile)
    print(f'{obs_id:30s} {percentage_long_ok:3.0f}% {percentage_ew_ok:3.0f}%', end='')

    if (np.sum(~bad_long)*arguments.longitude_resolution*arguments.downsample <
        arguments.minimum_coverage):
        print(' Skipped due to poor coverage')
        continue

    print(f' EW {ew_mean:6.3f} +/- {ew_std:6.3f}', end='')

    rd = arguments.ew_inner_radius - arguments.ring_radius
    rr = arguments.radius_resolution

    filtered_restr_bsm_img = restr_bsm_img #nd.median_filter(restr_bsm_img, (1,19))

    if arguments.compute_widths:
        widths1 = ma.zeros((len(longitudes), 2), dtype=float)
        widths1[:] = ma.masked
        widths2 = ma.zeros((len(longitudes), 2), dtype=float)
        widths2[:] = ma.masked
        widths3 = ma.zeros((len(longitudes), 2), dtype=float)
        widths3[:] = ma.masked

        for idx in range(len(longitudes)):
            if bad_long[idx]:
                continue
            # ret = max_range(restr_bsm_img[:,idx], (0.5,0.7,0.9),
            #                 None)
            ret = width_from_wings(filtered_restr_bsm_img[:,idx], (0.50, 0.75, 1,),
                                   width_brightness)
            widths1[idx] = (ret[0][0]*rr+rd, ret[0][1]*rr+rd)
            widths2[idx] = (ret[1][0]*rr+rd, ret[1][1]*rr+rd)
            widths3[idx] = (ret[2][0]*rr+rd, ret[2][1]*rr+rd)

        w1 = widths1[:,1]-widths1[:,0]
        w2 = widths2[:,1]-widths2[:,0]
        w3 = widths3[:,1]-widths3[:,0]
        w1_mean = ma.mean(w1)
        w1_std = ma.std(w1)
        w2_mean = ma.mean(w2)
        w2_std = ma.std(w2)
        w3_mean = ma.mean(w3)
        w3_std = ma.std(w3)
        print(f'  w1: {w1_mean:4.0f} +/- {w1_std:3.0f}', end='')
        print(f'  w2: {w2_mean:4.0f} +/- {w2_std:3.0f}', end='')
        print(f'  w3: {w3_mean:4.0f} +/- {w3_std:3.0f}', end='')

    print()

    if arguments.output_csv_filename:
        slice_size_in_longitudes = int(arguments.slice_size /
                                       arguments.longitude_resolution)
        if arguments.slice_size == 0:
            num_slices = 1
            slice_size_in_longitudes = len(longitudes)
        else:
            num_slices = len(longitudes) // slice_size_in_longitudes
        for slice_num in range(num_slices):
            slice_start = slice_num * slice_size_in_longitudes
            slice_end = (slice_num+1) * slice_size_in_longitudes
            slice_bad_long = bad_long[slice_start:slice_end]
            if np.all(slice_bad_long):
                continue
            slice_good_long = ~slice_bad_long
            slice_ETs = ETs[slice_start:slice_end][slice_good_long]
            slice_emission_angles = emission_angles[slice_start:slice_end][slice_good_long]
            slice_phase_angles = phase_angles[slice_start:slice_end][slice_good_long]
            slice_resolutions = resolutions[slice_start:slice_end][slice_good_long]
            slice_ew_profile = ew_profile[slice_start:slice_end][slice_good_long]
            slice_longitudes = longitudes[slice_start:slice_end][slice_good_long]
            if arguments.compute_widths:
                slice_w1 = w1[slice_start:slice_end][slice_good_long]
                slice_w2 = w2[slice_start:slice_end][slice_good_long]
                slice_w3 = w3[slice_start:slice_end][slice_good_long]

            slice_min_long = np.degrees(np.min(slice_longitudes))
            slice_max_long = np.degrees(np.max(slice_longitudes))

            slice_min_et = np.min(slice_ETs)
            slice_max_et = np.max(slice_ETs)
            slice_mean_et = np.mean(slice_ETs)
            slice_et_date = julian.iso_from_tai(julian.tai_from_tdb(slice_min_et))

            slice_min_em = np.min(slice_emission_angles)
            slice_max_em = np.max(slice_emission_angles)
            slice_mean_em = np.mean(slice_emission_angles)

            slice_min_ph = np.min(slice_phase_angles)
            slice_max_ph = np.max(slice_phase_angles)
            slice_mean_ph = np.mean(slice_phase_angles)

            slice_min_res = np.min(slice_resolutions)
            slice_max_res = np.max(slice_resolutions)
            slice_mean_res = np.mean(slice_resolutions)

            slice_ew_mean = np.mean(slice_ew_profile)
            slice_ew_std = np.std(slice_ew_profile)
            slice_ew_profile_mu = (slice_ew_profile *
                                   np.abs(np.cos(slice_emission_angles)))
            slice_ew_mean_mu = np.mean(slice_ew_profile_mu)
            slice_ew_std_mu = np.std(slice_ew_profile_mu)

            # if slice_ew_mean <= 0:
            #     print(obs_id, slice_num, 'EW Mean < 0')
            #     continue
            row = [obs_id, slice_num, np.sum(~slice_bad_long), slice_et_date,
                   np.round(slice_min_long, 2),
                   np.round(slice_max_long, 2),
                   np.round(slice_min_res, 3),
                   np.round(slice_max_res, 3),
                   np.round(slice_mean_res, 3),
                   np.round(np.degrees(slice_min_ph), 3),
                   np.round(np.degrees(slice_max_ph), 3),
                   np.round(np.degrees(slice_mean_ph), 3),
                   np.round(np.degrees(slice_min_em), 3),
                   np.round(np.degrees(slice_max_em), 3),
                   np.round(np.degrees(slice_mean_em), 3),
                   np.round(np.degrees(incidence_angle), 3),
                   percentage_ew_ok,
                   np.round(slice_ew_mean, 5),
                   np.round(slice_ew_std, 5),
                   np.round(slice_ew_mean_mu, 5),
                   np.round(slice_ew_std_mu, 5)]

            if arguments.compute_widths:
                row += [
                        np.round(np.mean(slice_w1), 3),
                        np.round(np.std(slice_w1), 3),
                        np.round(np.mean(slice_w2), 3),
                        np.round(np.std(slice_w2), 3),
                        np.round(np.mean(slice_w3), 3),
                        np.round(np.std(slice_w3), 3)]
            writer.writerow(row)

    if arguments.plot_results or arguments.save_plots:
        fig = plt.figure(figsize=(12, 8))
        plt.suptitle(obs_id)
        if arguments.compute_widths:
            ax = fig.add_subplot(311)
        else:
            ax = fig.add_subplot(111)
        plt.plot(np.degrees(longitudes), ew_profile)
        plt.xlim(0, 360)
        plt.ylabel('EW (km)')
        plt.xlabel('Longitude (degrees)')

        if arguments.compute_widths:
            gamma = 0.5
            blackpoint = max(np.min(restr_bsm_img[:, ~bad_long]), 0)
            whitepoint_ignore_frac = 0.995
            img_sorted = sorted(list(restr_bsm_img[:, ~bad_long].flatten()))
            whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                                whitepoint_ignore_frac),
                                            0, len(img_sorted)-1)]
            greyscale_img = np.floor((np.maximum(restr_bsm_img-blackpoint, 0)/
                                      (whitepoint-blackpoint))**gamma*256)
            greyscale_img = np.clip(greyscale_img, 0, 255)
            ax = fig.add_subplot(312)
            plt.imshow(greyscale_img[::-1, :],
                       extent=(0, 360,
                               arguments.ew_inner_radius - arguments.ring_radius,
                               arguments.ew_outer_radius - arguments.ring_radius),
                       aspect='auto',
                       cmap='gray', vmin=0, vmax=255)
            plt.plot(np.degrees(longitudes), widths1[:,0], color='cyan', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths1[:,1], color='cyan', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths2[:,0], color='orange', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths2[:,1], color='orange', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths3[:,0], color='red', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths3[:,1], color='red', lw=1, alpha=0.5)
            # plt.legend()
            plt.xlim(0, 360)
            plt.ylim(arguments.ew_inner_radius - arguments.ring_radius,
                     arguments.ew_outer_radius - arguments.ring_radius)
            plt.ylabel('Core offset (km)')
            plt.xlabel('Longitude (degrees)')
            ax = fig.add_subplot(313)
            plt.plot(np.degrees(longitudes), w1, color='cyan', label='50%')
            plt.plot(np.degrees(longitudes), w2, color='orange', label='75%')
            plt.plot(np.degrees(longitudes), w3, color='red', label='100%')
            plt.legend()
            plt.xlim(0, 360)
            plt.ylim(200, 1500)
            plt.ylabel('Ring width (km)')
            plt.xlabel('Longitude (degrees)')

        plt.tight_layout()
        if arguments.save_plots:
            if not os.path.exists('plots'):
                os.mkdir('plots')
            plt.savefig('plots/'+obs_id+'.png')
            fig.clear()
            plt.close()
            plt.cla()
            plt.clf()
        if arguments.plot_results:
            plt.show()

if arguments.output_csv_filename:
    csv_fp.close()
