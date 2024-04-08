################################################################################
# plot_offsets.py
#
# Plot the offsets for one mosaic.
################################################################################

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from nav.file import (img_to_offset_path,
                      read_offset_metadata)
import nav.logging_setup

from ring import ring_util

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['--ring-type', 'FMOVIE', 'ISS_029RF_FMOVIE001_VIMS']

parser = argparse.ArgumentParser()

ring_util.ring_add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

ring_util.ring_init(arguments)

nav.logging_setup.set_main_module_name('plot_offsets')

def plot_one_obsid(image_paths):
    fig, ax1 = plt.subplots(1, 1, figsize=(9,3))
    all_xs = []
    all_ys = []
    xs_by_winner = {}
    ys_by_winner = {}
    num_by_winner = {}

    for image_num, image_path in enumerate(image_paths):
        metadata = read_offset_metadata(image_path,
                                        arguments.instrument_host,
                                        'saturn')
        the_offset = metadata['offset']
        if the_offset is None:
            continue
        all_xs.append(the_offset[0])
        all_ys.append(the_offset[1])
        winner = metadata['offset_winner']
        if winner not in xs_by_winner:
            xs_by_winner[winner] = []
            ys_by_winner[winner] = []
            num_by_winner[winner] = []
        xs_by_winner[winner].append(the_offset[0])
        ys_by_winner[winner].append(the_offset[1])
        num_by_winner[winner].append(image_num)

    # plt.plot(all_xs, '--', color='black', label='X Offset')
    # plt.plot(all_ys, '-', color='black', label='Y Offset')
    # plt.plot(all_xs, 's', mec='black', mfc='black', label='X Offset')
    # plt.plot(all_ys, 's', mec='black', mfc='none', label='Y Offset')

    symbol_by_winner = {
        'STARS': ('*', 10),
        'MODEL': ('o', 6),
    }
    label_by_winner = {
        'STARS': 'Stars',
        'MODEL': 'Main Ring'
    }
    pxs = []
    pys = []
    for winner in symbol_by_winner:
        print(winner, len(num_by_winner[winner]))
        symbol = symbol_by_winner[winner]
        label = label_by_winner[winner]
        px, = ax1.plot(num_by_winner[winner], xs_by_winner[winner],
                       symbol[0], ms=symbol[1], mec='black', mfc='none')
        py, = ax1.plot(num_by_winner[winner], ys_by_winner[winner],
                       symbol[0], ms=symbol[1], mec='black', mfc='black')
        pxs.append(px)
        pys.append(py)
    ax1.legend([tuple(pxs), tuple(pys)], ['X Offset', 'Y Offset'],
               handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='upper center')
    plt.xlabel('Image number')
    plt.ylabel('Offset (Cassini ISS NAC pixels)')
    plt.tight_layout()
    plt.savefig('figure.png')
    plt.show()


def plot_all_obsids():
    image_paths = []
    prev_obsid = None
    for obsid, image_name, image_path in ring_util.ring_enumerate_files(arguments):
        if obsid != prev_obsid:
            if len(image_paths):
                plot_one_obsid(image_paths)
            prev_obsid = obsid
            image_paths = []
        image_paths.append(image_path)
    if len(image_paths):
        plot_one_obsid(image_paths)

plot_all_obsids()
