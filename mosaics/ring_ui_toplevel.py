################################################################################
# ring_ui_toplevel.py
#
# The main top-level GUI for ring mosaic making.
################################################################################

import argparse
import os
import subprocess
import sys

from tkinter import *
import tkinter.messagebox
from imgdisp import IntegerEntry, FloatEntry, ScrolledList

from nav.file import (img_to_offset_path,
                      read_offset_metadata)
import ring.ring_util as ring_util


PYTHON_EXE = sys.executable

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['--ring-type', 'FMOVIE', '--all-obsid']

parser = argparse.ArgumentParser()

ring_util.ring_add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

ring_util.ring_init(arguments)


# Storage for offset+reprojection status
class OffRepStatus:
    def __init__(self):
        self.obsid = None
        self.image_name = None
        self.image_path = None
        self.offset_path = None
        self.repro_path = None

        # 'EE' = Offset error
        # '--' = No offset file
        # Automatic status:
        #   ' ' = No auto offset
        #   'L' = Model winner, no stars available
        #   'l' = Model winner, stars available
        #   'S' = Stars winner
        #   '?' = Other winner
        #   '@' = Unknown issue
        # Manual status:
        #   ' ' = No manual offset
        #   'M' = Manual offset present
        self.offset_status = None

        self.auto_offset = None
        self.manual_offset = None
        self.offset_mtime = None

        # 'X' = No repro file
        # 'R' = OK repro file up-to-date
        # 'r' = OK repro file but missing offset file
        # 'D' = OK repro file but out of date compared to offset
        self.repro_status = None

        self.repro_mtime = None

# Storage for user input to the GUI
class GUIData:
    def __init__(self):
        self.obsid_db = None
        self.obsid_selection = None
        self.img_selection = None
        self.ring_radius = None
        self.entry_radius_inner = None
        self.entry_radius_outer = None
        self.entry_radius_resolution = None
        self.entry_longitude_resolution = None
        self.entry_radial_zoom_amount = None
        self.entry_longitude_zoom_amount = None


################################################################################
#
# SUPPORT ROUTINES
#
################################################################################

#
# Iterate through ALL images for ALL obsids and find their status
# Returns a dictionary of observation IDs, each entry of which is a list of
# OffRepStatus plus status for the mosaic and background files
#
def get_file_status(guidata, obsid=None, do_stat=True):
    if obsid is None:
        obsid_db = {}
    else:
        # We're only going to replace one entry
        obsid_db = guidata.obsid_db

    ring_radius = guidata.entry_ring_radius.value()
    radius_inner = guidata.entry_radius_inner.value()
    radius_outer = guidata.entry_radius_outer.value()
    radius_resolution = guidata.entry_radius_resolution.value()
    longitude_resolution = guidata.entry_longitude_resolution.value()
    radial_zoom_amount = guidata.entry_radial_zoom_amount.value()
    longitude_zoom_amount = guidata.entry_longitude_zoom_amount.value()

    cur_obsid = None
    status_list = []
    max_repro_mtime = 0

    for obsid, image_name, image_path in ring_util.ring_enumerate_files(
                                                            arguments,
                                                            force_obsid=obsid):
        offrepstatus = OffRepStatus()
        offrepstatus.obsid = obsid
        offrepstatus.image_name = image_name
        offrepstatus.image_path = image_path

        offrepstatus.offset_path = img_to_offset_path(
                            image_path,
                            instrument_host=arguments.instrument_host)
        offrepstatus.repro_path = ring_util.img_to_repro_path_spec(
                    ring_radius,
                    radius_inner, radius_outer,
                    radius_resolution, longitude_resolution,
                    radial_zoom_amount, longitude_zoom_amount,
                    image_path, arguments.instrument_host)
        if not do_stat:
            offrepstatus.offset_status = '??'
            offrepstatus.offset_mtime = 1e38
            offrepstatus.repro_status = '?'
            offrepstatus.repro_mtime = 1e37
        else:
            offrepstatus.offset_status = ''
            if os.path.exists(offrepstatus.offset_path):
                offrepstatus.offset_mtime = (os.stat(offrepstatus.offset_path).
                                             st_mtime)
            else:
                offrepstatus.offset_status = '--'
                offrepstatus.offset_mtime = 1e38

            if os.path.exists(offrepstatus.repro_path):
                offrepstatus.repro_mtime = os.stat(
                                       offrepstatus.repro_path).st_mtime
                if offrepstatus.repro_mtime < 1676416981:
                    print(obsid, image_name, offrepstatus.repro_mtime)
                if offrepstatus.offset_mtime == 1e38:
                    offrepstatus.repro_status = 'r'
                elif offrepstatus.offset_mtime > offrepstatus.repro_mtime:
                    offrepstatus.repro_status = 'D'
                else:
                    offrepstatus.repro_status = 'R'
            else:
                offrepstatus.repro_status = 'X'
                offrepstatus.repro_mtime = 1e37

        if cur_obsid is None:
            cur_obsid = obsid
        if cur_obsid != obsid:
            if len(status_list) != 0:
                prefix_mosaic = get_mosaic_status(cur_obsid, max_repro_mtime)
                prefix_bkgnd = get_bkgnd_status(cur_obsid)
                obsid_db[cur_obsid] = (prefix_mosaic, prefix_bkgnd, status_list)
            status_list = []
            max_repro_mtime = 0
            cur_obsid = obsid

        if offrepstatus.repro_mtime < 1e35:
            max_repro_mtime = max(max_repro_mtime, offrepstatus.repro_mtime)
        status_list.append(offrepstatus)

    if len(status_list) != 0:
        prefix_mosaic = get_mosaic_status(cur_obsid, max_repro_mtime)
        prefix_bkgnd = get_bkgnd_status(cur_obsid)
        obsid_db[cur_obsid] = (prefix_mosaic, prefix_bkgnd, status_list)

    return obsid_db

#
# Set the offset status flags for all images in a single obsid if there is no
# status already present. This does not update existing status.
#
def cache_offset_status_for_obsid(obsid_db, obsid):
    if obsid is None:
        return
    prefix_mosaic, prefix_bkgnd, status_list = obsid_db[obsid]
    for offrepstatus in status_list:
        read_one_offset_status(offrepstatus)

#
# Update the offset and repro status for a single OffRepStatus
#
def read_one_offset_status(offrepstatus):
    if offrepstatus.offset_status != '':
        return

    offrepstatus.offset_metadata = read_offset_metadata(
                                                 offrepstatus.image_path,
                                                 arguments.instrument_host,
                                                 'saturn',
                                                 overlay=False)

    auto_status = ' '
    man_status = ' '

    status = offrepstatus.offset_metadata['status']
    if status != 'ok':
        auto_status = 'E'
    else:
        offrepstatus.auto_offset = offrepstatus.offset_metadata['offset']
    if 'manual_offset' in offrepstatus.offset_metadata:
        offrepstatus.manual_offset = offrepstatus.offset_metadata[
                                                  'manual_offset']
    else:
        offrepstatus.manual_offset = None

    if offrepstatus.auto_offset is not None:
        try:
            if offrepstatus.offset_metadata['offset_winner'] == 'MODEL':
                if offrepstatus.offset_metadata['stars_offset'] is not None:
                    auto_status = 'l'
                else:
                    auto_status = 'L'
            elif offrepstatus.offset_metadata['offset_winner'] == 'STARS':
                auto_status = 'S'
            else:
                auto_status = '?'
        except:
            auto_status = '@'
    if offrepstatus.manual_offset is not None:
        man_status = 'M'
    offrepstatus.offset_status = auto_status + man_status
    if offrepstatus.offset_status == '  ':
        offrepstatus.offset_status = 'XX'

#
# Get the mosaic status for a single obsid
# 'X' = No mosaic file
# 'D' = Mosaic file but out of date compared to repro files
# 'M' = Mosaic file up to date
#
def get_mosaic_status(cur_obsid, max_repro_mtime):
    ring_radius = guidata.entry_ring_radius.value()
    radius_inner = guidata.entry_radius_inner.value()
    radius_outer = guidata.entry_radius_outer.value()
    radius_resolution = guidata.entry_radius_resolution.value()
    longitude_resolution = guidata.entry_longitude_resolution.value()
    radial_zoom_amount = guidata.entry_radial_zoom_amount.value()
    longitude_zoom_amount = guidata.entry_longitude_zoom_amount.value()

    (data_path, metadata_path) = ring_util.mosaic_paths_spec(ring_radius,
                                             radius_inner,
                                             radius_outer,
                                             radius_resolution,
                                             longitude_resolution,
                                             radial_zoom_amount,
                                             longitude_zoom_amount,
                                             cur_obsid,
                                             arguments.ring_type)
    if (not os.path.exists(data_path+'.npy') or
        not os.path.exists(metadata_path)):
        prefix = 'X'
    elif (os.stat(data_path+'.npy').st_mtime < max_repro_mtime or
          os.stat(metadata_path).st_mtime < max_repro_mtime):
        prefix = 'D'
    else:
        prefix = 'M'
    return prefix

#
# Get the background status for a single obsid
# 'X' = No background file
# 'D' = Background file but out of date compared to mosaic file
# 'B' = Background file up to date compared to mosaic file
# 'b' = Background file but no mosaic file
#
def get_bkgnd_status(cur_obsid):
    ring_radius = guidata.entry_ring_radius.value()
    radius_inner = guidata.entry_radius_inner.value()
    radius_outer = guidata.entry_radius_outer.value()
    radius_resolution = guidata.entry_radius_resolution.value()
    longitude_resolution = guidata.entry_longitude_resolution.value()
    radial_zoom_amount = guidata.entry_radial_zoom_amount.value()
    longitude_zoom_amount = guidata.entry_longitude_zoom_amount.value()

    (data_path, metadata_path) = ring_util.mosaic_paths_spec(
                                                     ring_radius,
                                                     radius_inner,
                                                     radius_outer,
                                                     radius_resolution,
                                                     longitude_resolution,
                                                     radial_zoom_amount,
                                                     longitude_zoom_amount,
                                                     cur_obsid,
                                                     arguments.ring_type)

    if (not os.path.exists(data_path+'.npy') or
        not os.path.exists(metadata_path)):
        max_mosaic_mtime = 0
    else:
        max_mosaic_mtime = max(os.stat(data_path+'.npy').st_mtime,
                               os.stat(metadata_path).st_mtime)

    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
     bkgnd_model_filename,
     bkgnd_metadata_filename) = ring_util.bkgnd_paths_spec(
                                                   ring_radius,
                                                   radius_inner, radius_outer,
                                                   radius_resolution,
                                                   longitude_resolution,
                                                   radial_zoom_amount,
                                                   longitude_zoom_amount,
                                                   cur_obsid,
                                                   arguments.ring_type)
    if (not os.path.exists(bkgnd_model_filename) or
        not os.path.exists(bkgnd_metadata_filename)):
        prefix = 'X'
    elif (os.stat(bkgnd_model_filename).st_mtime < max_mosaic_mtime or
          os.stat(bkgnd_metadata_filename).st_mtime < max_mosaic_mtime):
        prefix = 'D'
    else:
        if max_mosaic_mtime == 0:
            prefix = 'b'
        else:
            prefix = 'B'
    return prefix

#
# Return the list of mosaic+background status and OBSIDs to populate the main
# list box
#
def mosaic_background_status_names(obsid_db):
    obsid_names = []
    for key in sorted(obsid_db.keys()):
        obsid_names.append(obsid_db[key][0]+' '+obsid_db[key][1]+' '+key)
    return obsid_names

#
# Go through the list entries one at a time and replce, insert, or delete items
# as appropriate. This assumes the lists are in alphabetical order!
#
def update_one_list(listbox, new_list_entries, char_skip=0):
    gui_list_entry_num = 0
    new_list_entry_num = 0
    while new_list_entry_num < len(new_list_entries):
        try:
            gui_list_entry = listbox[gui_list_entry_num][char_skip:]
        except IndexError:
            gui_list_entry = None
        new_list_entry = new_list_entries[new_list_entry_num][char_skip:]
        if gui_list_entry == new_list_entry:
            # The list entry hasn't changed - see if the full entry has changed
            if listbox[gui_list_entry_num] != new_list_entries[
                                                       new_list_entry_num]:
                # The details have changed, so delete and reinsert
                listbox.delete(gui_list_entry_num)
                listbox.insert(gui_list_entry_num,
                               new_list_entries[new_list_entry_num])
            gui_list_entry_num = min(gui_list_entry_num+1, listbox.count())
            new_list_entry_num += 1
            continue
        if gui_list_entry is not None and gui_list_entry < new_list_entry:
            # An entry got deleted
            listbox.delete(gui_list_entry_num)
            continue
        # An entry got inserted
        listbox.insert(gui_list_entry_num,
                       new_list_entries[new_list_entry_num])
        gui_list_entry_num = min(gui_list_entry_num+1, listbox.count())
        new_list_entry_num += 1
    while gui_list_entry_num < listbox.count():
        listbox.delete(gui_list_entry_num)


################################################################################
#
# OBSID / IMAGE LIST FOR OFFSET/REPROJECTION
#
################################################################################

#
# Make command-line arguments for subprocesses
#
def cmdline_arguments(guidata, subprocesses=False):
    ret = ring_util.ring_basic_cmd_line(arguments)
    ring_radius = guidata.entry_ring_radius.value()
    radius_inner = guidata.entry_radius_inner.value()
    radius_outer = guidata.entry_radius_outer.value()
    radius_resolution = guidata.entry_radius_resolution.value()
    longitude_resolution = guidata.entry_longitude_resolution.value()
    radial_zoom_amount = guidata.entry_radial_zoom_amount.value()
    longitude_zoom_amount = guidata.entry_longitude_zoom_amount.value()

    ret += ['--ring-type', str(arguments.ring_type),
            '--ring-radius', str(ring_radius),
            '--radius-inner-delta', str(radius_inner),
            '--radius-outer-delta', str(radius_outer),
            '--radius-resolution', '%.3f'%radius_resolution,
            '--longitude-resolution', '%.3f'%longitude_resolution,
            '--radial-zoom-amount', '%d'%radial_zoom_amount,
            '--longitude-zoom-amount', '%d'%longitude_zoom_amount,
            '--instrument-host', arguments.instrument_host]
    if subprocesses:
        ret += ['--max-subprocesses', '2']

    return ret

#
# Button press on the obsid list - update image list with existing data
#
def offrep_obsid_list_buttonrelease_handler(event, guidata):
    obsid_selections = guidata.listbox_obsid.listbox.curselection()
    if len(obsid_selections) == 0:
        return
    guidata.obsid_selection = guidata.listbox_obsid[
                                            int(obsid_selections[0])][4:]
    offrep_update_img_list(guidata)

#
# Update image list based on current obsid
#
def offrep_update_img_list(guidata):
    cache_offset_status_for_obsid(guidata.obsid_db, guidata.obsid_selection)
    guidata.img_selection = None
    guidata.cur_img_list = []
    if guidata.obsid_selection is not None:
        for data in guidata.obsid_db[guidata.obsid_selection][2]:
            img_string = '[%2s] [%s] %s' % (data.offset_status,
                                            data.repro_status,
                                            data.image_name)
            guidata.cur_img_list.append(img_string)
    update_one_list(guidata.listbox_img, guidata.cur_img_list, 16)
    if guidata.obsid_selection is None:
        guidata.label_images.config(text='Images:')
    else:
        guidata.label_images.config(text=guidata.obsid_selection + ' Images:')

#
# Button press on the image list - pop up ring_reproject on image
#
def offrep_img_list_buttonrelease_handler(event, guidata):
    img_selections = guidata.listbox_img.listbox.curselection()
    if guidata.obsid_selection is None:
        return
    guidata.img_selection = guidata.obsid_db[
                         guidata.obsid_selection][2][int(img_selections[0])]
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--display-offset-reproject',
                      '--no-auto-offset',
                      '--no-reproject',
                      '--image-log-console-level', 'debug',
                      guidata.obsid_selection + '/' +
                      guidata.img_selection.image_name] +
                      cmdline_arguments(guidata))

#
# Refresh Files button
#
def offrep_refresh_button_handler(guidata):
    if guidata.obsid_selection is None:
        return
    guidata.obsid_db = get_file_status(guidata, guidata.obsid_selection)
    update_one_list(guidata.listbox_obsid,
                    mosaic_background_status_names(guidata.obsid_db))
    offrep_update_img_list(guidata)

#
# Refresh All Files button
#
def offrep_refresh_all_button_handler(guidata):
    guidata.obsid_db = get_file_status(guidata)
    update_one_list(guidata.listbox_obsid,
                    mosaic_background_status_names(guidata.obsid_db))
    offrep_update_img_list(guidata)

#
# Display Mosaic button
#
def offrep_display_mosaic_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Display Mosaic',
                                     'No current OBSID selection')
        return
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_MOSAIC_PY,
                      '--display-mosaic', '--no-mosaic',
                      guidata.obsid_selection] +
                      cmdline_arguments(guidata))

#
# Display Background button
#
def offrep_display_bkgnd_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Display Background',
                                     'No current OBSID selection')
        return
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_BKGND_PY,
                      '--display-bkgnd',
                      '--no-bkgnd', guidata.obsid_selection] +
                      cmdline_arguments(guidata))

#
# Update Offsets button
#
def offrep_update_offsets_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Update Offsets',
                                     'No current OBSID selection')
        return
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--verbose',
                      '--no-reproject',
                      guidata.obsid_selection] +
                     cmdline_arguments(guidata, subprocesses=True))

#
# Update All Offsets button
#
def offrep_update_all_offsets_button_handler(guidata):
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--verbose',
                      '--no-reproject', '--all-obsid'] +
                     cmdline_arguments(guidata, subprocesses=True))

#
# Force Update Offsets button
#
def offrep_force_update_offsets_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Force Update Offsets',
                               'No current OBSID selection')
        return
    if not tkinter.messagebox.askyesno('Force Update Offsets',
                                 'Are you sure you want to do a forced update'+
                                 ' on ALL offsets in this OBSID?'):
        return
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--verbose',
                      '--recompute-auto-offset',
                      '--no-reproject', guidata.obsid_selection] +
                     cmdline_arguments(guidata, subprocesses=True))

#
# Force Update All Offsets button
#
def offrep_force_update_all_offsets_button_handler(guidata):
    if not tkinter.messagebox.askyesno('Force Update All Offsets',
                                 'Are you sure you want to do a forced update'+
                                 ' on ALL offsets?'):
        return
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--verbose',
                      '--recompute-auto-offset',
                      '--no-reproject', '--all-obsid'] +
                     cmdline_arguments(guidata, subprocesses=True))

#
# Update Reprojections button
#
def offrep_update_reprojects_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Update Reprojections',
                               'No current OBSID selection')
        return
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--verbose',
                      '--no-auto-offset', guidata.obsid_selection] +
                     cmdline_arguments(guidata, subprocesses=True))

#
# Update All Reprojections button
#
def offrep_update_all_reprojects_button_handler(guidata):
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--verbose',
                      '--no-auto-offset', '--all-obsid'] +
                     cmdline_arguments(guidata, subprocesses=True))

#
# Force Update Reprojections button
#
def offrep_force_update_reprojects_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Force Update Reprojections',
                               'No current OBSID selection')
        return
    if not tkinter.messagebox.askyesno('Force Update Reprojections',
                                 'Are you sure you want to do a forced update'+
                                 ' on ALL reprojections in this OBSID?'):
        return
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--verbose',
                      '--recompute-reproject',
                      '--no-auto-offset', guidata.obsid_selection] +
                     cmdline_arguments(guidata, subprocesses=True))

#
# Force Update All Reprojections button
#
def offrep_force_update_all_reprojects_button_handler(guidata):
    if not tkinter.messagebox.askyesno('Force Update All Reprojections',
                                 'Are you sure you want to do a forced update'+
                                 ' on ALL reprojections?'):
        return
    subprocess.Popen([PYTHON_EXE,
                      ring_util.RING_REPROJECT_PY,
                      '--verbose',
                      '--recompute-reproject',
                      '--no-auto-offset', '-a'] +
                     cmdline_arguments(guidata, subprocesses=True))

#
# Update Mosaic button
#
def offrep_update_mosaic_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Update Mosaic',
                                     'No current OBSID selection')
        return
    subprocess.Popen([PYTHON_EXE, ring_util.RING_MOSAIC_PY,
                      guidata.obsid_selection] +
                     cmdline_arguments(guidata))

#
# Update All Mosaics button
#
def offrep_update_all_mosaics_button_handler(guidata):
    subprocess.Popen([PYTHON_EXE, ring_util.RING_MOSAIC_PY, '--all-obsid'] +
                     cmdline_arguments(guidata))

#
# Force Update Mosaic button
#
def offrep_force_update_mosaic_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Force Update Mosaic',
                                     'No current OBSID selection')
        return
    subprocess.Popen([PYTHON_EXE, ring_util.RING_MOSAIC_PY,
                      '--recompute-mosaic',
                      guidata.obsid_selection] +
                      cmdline_arguments(guidata))

#
# Force Update All Mosaics button
#
def offrep_force_update_all_mosaics_button_handler(guidata):
    if not tkinter.messagebox.askyesno('Force Update All Mosaics',
                                 'Are you sure you want to do a forced update'+
                                 ' on ALL mosaics?'):
        return
    subprocess.Popen([PYTHON_EXE, ring_util.RING_MOSAIC_PY,
                      '--all-obsid', '--recompute-mosaic'] +
                     cmdline_arguments(guidata))

#
# Update Background button
#
def offrep_update_bkgnd_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Update Background',
                               'No current OBSID selection')
        return
    subprocess.Popen([PYTHON_EXE, ring_util.RING_BKGND_PY,
                      guidata.obsid_selection] +
                     cmdline_arguments(guidata))

#
# Update All Backgrounds button
#
def offrep_update_all_bkgnds_button_handler(guidata):
    subprocess.Popen([PYTHON_EXE, ring_util.RING_BKGND_PY,
                      '--all-obsid'] +
                     cmdline_arguments(guidata))

#
# Force Update Background button
#
def offrep_force_update_bkgnd_button_handler(guidata):
    if guidata.obsid_selection is None:
        tkinter.messagebox.showerror('Force Update Background',
                               'No current OBSID selection')
        return
    subprocess.Popen([PYTHON_EXE, ring_util.RING_BKGND_PY,
                      '--recompute-bkgnd',
                      guidata.obsid_selection] +
                      cmdline_arguments(guidata))

#
# Force Update All Backgrounds button
#
def offrep_force_update_all_bkgnds_button_handler(guidata):
    if not tkinter.messagebox.askyesno('Force Update All Backgrounds',
                                 'Are you sure you want to do a forced update'+
                                 ' on ALL backgrounds?'):
        return
    subprocess.Popen([PYTHON_EXE, ring_util.RING_BKGND_PY,
                      '--all-obsid', '--recompute-bkgnd'] +
                     cmdline_arguments(guidata))

#
# Create obsid list and image list
#
def offrep_setup_obs_lists(guidata, imglist=False):
    guidata.frame_obsid_img = Frame()
    label = Label(guidata.frame_obsid_img, text='Observation IDs:')
    label.grid(row=0, column=0)
    guidata.listbox_obsid = ScrolledList(guidata.frame_obsid_img, width=35,
                                         selectmode=BROWSE,
                                         font=('Courier', 10))
    guidata.listbox_obsid.listbox.bind("<ButtonRelease-1>",
            lambda event, guidata=guidata:
            offrep_obsid_list_buttonrelease_handler(event, guidata))
    guidata.listbox_obsid.grid(row=1, column=0)

    ### Controls Frame

    frame_controls = Frame(guidata.frame_obsid_img)
    controls_row = 0

    button_refresh = Button(frame_controls, text='Refresh Files',
                            command=lambda guidata=guidata:
                                    offrep_refresh_button_handler(guidata))
    button_refresh.grid(row=controls_row, column=0)

    button_refresh = Button(frame_controls, text='Refresh All Files',
                            command=lambda guidata=guidata:
                                    offrep_refresh_all_button_handler(guidata))
    button_refresh.grid(row=controls_row, column=1)

    button_display_mosaic = Button(frame_controls, text='Display Mosaic',
                                   command=lambda guidata=guidata:
                                offrep_display_mosaic_button_handler(guidata))
    button_display_mosaic.grid(row=controls_row, column=2)

    button_display_bkgnd = Button(frame_controls, text='Display Background',
                                  command=lambda guidata=guidata:
                                  offrep_display_bkgnd_button_handler(guidata))
    button_display_bkgnd.grid(row=controls_row, column=3)

    controls_row += 1

    # Specs for reprojection
    frame_reprojection = Frame(frame_controls)
    label = Label(frame_reprojection, text='Ring radius:')
    label.pack(side=LEFT)
    guidata.entry_ring_radius = IntegerEntry(
                                      frame_reprojection,
                                      value=arguments.ring_radius)
    guidata.entry_ring_radius.pack(side=LEFT)
    label = Label(frame_reprojection, text='Radius inner:')
    label.pack(side=LEFT)
    guidata.entry_radius_inner = IntegerEntry(
                                      frame_reprojection,
                                      value=arguments.radius_inner_delta)
    guidata.entry_radius_inner.pack(side=LEFT)
    label = Label(frame_reprojection, text='Radius outer:')
    label.pack(side=LEFT)
    guidata.entry_radius_outer = IntegerEntry(
                                      frame_reprojection,
                                      value=arguments.radius_outer_delta)
    guidata.entry_radius_outer.pack(side=LEFT)
    frame_reprojection.grid(row=controls_row, column=0, columnspan=5,
                            sticky=W)
    controls_row += 1

    frame_reprojection2 = Frame(frame_controls)
    label = Label(frame_reprojection2, text='Radial res:')
    label.pack(side=LEFT)
    guidata.entry_radius_resolution = FloatEntry(
                                         frame_reprojection2,
                                         value=arguments.radius_resolution)
    guidata.entry_radius_resolution.pack(side=LEFT)
    label = Label(frame_reprojection2, text='Longitude res:')
    label.pack(side=LEFT)
    guidata.entry_longitude_resolution = FloatEntry(
                                        frame_reprojection2,
                                        value=arguments.longitude_resolution)
    guidata.entry_longitude_resolution.pack(side=LEFT)
    frame_reprojection2.grid(row=controls_row, column=0, columnspan=5,
                             sticky=W)
    controls_row += 1

    frame_reprojection2a = Frame(frame_controls)
    label = Label(frame_reprojection2a, text='Radial zoom:')
    label.pack(side=LEFT)
    guidata.entry_radial_zoom_amount = IntegerEntry(
                                        frame_reprojection2a,
                                        value=arguments.radial_zoom_amount)
    guidata.entry_radial_zoom_amount.pack(side=LEFT)
    label = Label(frame_reprojection2a, text='Longitude zoom:')
    label.pack(side=LEFT)
    guidata.entry_longitude_zoom_amount = IntegerEntry(
                                        frame_reprojection2a,
                                        value=arguments.longitude_zoom_amount)
    guidata.entry_longitude_zoom_amount.pack(side=LEFT)
    frame_reprojection2a.grid(row=controls_row, column=0, columnspan=5,
                              sticky=W)
    controls_row += 1

    frame_reprojection3 = Frame(frame_controls)

    # Controls for offset
    button_update_offsets = Button(frame_controls,
                                   text='Update Offsets',
                                   command=lambda guidata=guidata:
                    offrep_update_offsets_button_handler(guidata))
    button_update_offsets.grid(row=controls_row, column=0)

    button_update_all_offsets = Button(frame_controls,
                                       text='Update All Offsets',
                                       command=lambda guidata=guidata:
                    offrep_update_all_offsets_button_handler(guidata))
    button_update_all_offsets.grid(row=controls_row, column=1)

    button_force_update_offsets = Button(frame_controls,
                                         text='Force Update Offsets',
                                         command=lambda guidata=guidata:
                    offrep_force_update_offsets_button_handler(guidata))
    button_force_update_offsets.grid(row=controls_row, column=2)

    button_force_update_all_offsets = Button(frame_controls,
                                             text='Force Update All Offsets',
                                             command=lambda guidata=guidata:
                    offrep_force_update_all_offsets_button_handler(guidata))
    button_force_update_all_offsets.grid(row=controls_row, column=3)
    controls_row += 1

    # Controls for reprojection
    button_update_reprojects = Button(frame_controls,
                                      text='Update Reprojections',
                                      command=lambda guidata=guidata:
                    offrep_update_reprojects_button_handler(guidata))
    button_update_reprojects.grid(row=controls_row, column=0)

    button_update_all_reprojects = Button(frame_controls,
                                          text='Update All Reprojections',
                                          command=lambda guidata=guidata:
                    offrep_update_all_reprojects_button_handler(guidata))
    button_update_all_reprojects.grid(row=controls_row, column=1)

    button_force_update_reprojects = Button(frame_controls,
                                            text='Force Update Reprojections',
                                            command=lambda guidata=guidata:
                    offrep_force_update_reprojects_button_handler(guidata))
    button_force_update_reprojects.grid(row=controls_row, column=2)

    button_force_update_all_reprojects = Button(frame_controls,
                                        text='Force Update All Reprojections',
                                        command=lambda guidata=guidata:
                    offrep_force_update_all_reprojects_button_handler(guidata))
    button_force_update_all_reprojects.grid(row=controls_row, column=3)
    controls_row += 1

    # Controls for mosaics
    button_update_mosaic = Button(frame_controls, text='Update Mosaic',
                                  command=lambda guidata=guidata:
                                  offrep_update_mosaic_button_handler(guidata))
    button_update_mosaic.grid(row=controls_row, column=0)

    button_update_all_mosaics = Button(frame_controls, text='Update All Mosaics',
                                       command=lambda guidata=guidata:
                    offrep_update_all_mosaics_button_handler(guidata))
    button_update_all_mosaics.grid(row=controls_row, column=1)

    button_update_mosaic = Button(frame_controls, text='Force Update Mosaic',
                                  command=lambda guidata=guidata:
                    offrep_force_update_mosaic_button_handler(guidata))
    button_update_mosaic.grid(row=controls_row, column=2)

    button_update_all_mosaics = Button(frame_controls,
                                       text='Force Update All Mosaics',
                                       command=lambda guidata=guidata:
                    offrep_force_update_all_mosaics_button_handler(guidata))
    button_update_all_mosaics.grid(row=controls_row, column=3)
    controls_row += 1

    # Controls for mosaics
    button_update_bkgnd = Button(frame_controls, text='Update Bkgnd',
                                 command=lambda guidata=guidata:
                    offrep_update_bkgnd_button_handler(guidata))
    button_update_bkgnd .grid(row=controls_row, column=0)

    button_update_all_bkgnds = Button(frame_controls, text='Update All Bkgnds',
                                      command=lambda guidata=guidata:
                    offrep_update_all_bkgnds_button_handler(guidata))
    button_update_all_bkgnds.grid(row=controls_row, column=1)

    button_update_bkgnd = Button(frame_controls, text='Force Update Bkgnd',
                                 command=lambda guidata=guidata:
                    offrep_force_update_bkgnd_button_handler(guidata))
    button_update_bkgnd.grid(row=controls_row, column=2)

    button_update_all_bkgnds = Button(frame_controls, text='Force Update All Bkgnds',
                                      command=lambda guidata=guidata:
                    offrep_force_update_all_bkgnds_button_handler(guidata))
    button_update_all_bkgnds.grid(row=controls_row, column=3)

    ###

    frame_controls.grid(row=2, column=0, columnspan=2)

    if imglist:
        guidata.label_images = Label(guidata.frame_obsid_img, text='Images:')
        guidata.label_images.grid(row=0, column=1)
        guidata.listbox_img = ScrolledList(guidata.frame_obsid_img, width=30,
                                           font=('Courier', 10))
        guidata.listbox_img.listbox.bind("<ButtonRelease-1>",
                lambda event, guidata=guidata:
                offrep_img_list_buttonrelease_handler(event, guidata))
        guidata.listbox_img.grid(row=1, column=1)
        guidata.cur_img_list = None

    guidata.frame_obsid_img.pack()

#
# Update the obsid and image lists
#
def offrep_update_obs_lists(guidata):
    update_one_list(guidata.listbox_obsid,
                    mosaic_background_status_names(guidata.obsid_db))
    if guidata.cur_img_list is not None:
        update_one_list(guidata.listbox_img, guidata.cur_img_list, 16)

###############################################

guidata = GUIData()

# Create the toplevel - otherwise creating variables below crashes
toplevel = Tk()

offrep_setup_obs_lists(guidata, imglist=True)

guidata.obsid_db = get_file_status(guidata, do_stat=True)

offrep_update_obs_lists(guidata)

mainloop()
