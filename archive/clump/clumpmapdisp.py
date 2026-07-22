'''
Created on Jul 31, 2012

@author: rfrench
'''

from tkinter import *
from PIL import ImageTk
from PIL import Image
import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as interp
#import ringutil
#ringutil.ComputeLongitudeShift
import cwt
import clump_util
from clump_util import ClumpData, ClumpDBEntry, ClumpChainData

ClumpData.userattr += ['canvas_obj', 'selected', 'highlighted']
ClumpChainData.userattr += ['selected', 'highlighted', 'chain_link_obj_list', 'straight_obj']

poster = False
class ClumpMapDisp(Frame):
    """
    [Values] are calculated from others.

    The main canvas has, from left to right:

        Left margin                    self.left_margin
        Vertical axis text             self.vert_axis_text_width
        Vertical axis text-ticks gap   self.vert_axis_text_gap_width
        Vertical axis ticks            self.vert_axis_ticks_width
        Vertical axis line width       self.vert_axis_line_width
        [Start of main drawing area]   self.drawing_area_x_start
        Drawing area
        [Drawing area width]           self.drawing_area_width
        [End of main drawing area]     self.drawing_area_x_end
        Right margin                   self.right_margin

    The main canvas has, from top to bottom:

        Top margin                     self.top_margin
        [Start of main drawing area]   self.drawing_area_y_start
        Drawing area
        [Drawing area height]          self.drawing_area_height
        [End of main drawing area]     self.drawing_area_y_end
        Horizontal axis line           self.horiz_axis_line_width
        Horizontal axis ticks          self.horiz_axis_ticks_height
        Horizontal axis text-ticks gap self.horiz_axis_text_gap_height
        Horizontal axis text           self.horiz_axis_text_height
        Bottom margin                  self.bottom_margin

    """


    def __init__(self, clump_db, chain_list=[], parent=None, canvas_size=None):
        '''Inputs:

        overlay_list - An optional list of 3-D (height x width x 3) color overlays, one per image
        parent       - The parent Tkinter widget
        canvas_size  - The size of the Canvas widget; defaults to XXX'''

        Frame.__init__(self, parent)

        if canvas_size == None:
            if poster:
                canvas_size = (1600, 635) #Poster size
            else:
                canvas_size = (1280, 768)
#

        self.canvas_size_x, self.canvas_size_y = canvas_size

        self.canvas_frame = Frame(self) # So we can have a canvas and controls
        if poster:
            self.canvas = Canvas(self.canvas_frame, width=self.canvas_size_x, height=self.canvas_size_y,
                             bg='#fef9f5', cursor='crosshair')
        else:
            self.canvas = Canvas(self.canvas_frame, width=self.canvas_size_x, height=self.canvas_size_y,
                                 bg='black', cursor='crosshair')
        self.canvas.grid(row=0, column=0, sticky=NW)
        #background color = '#fef9f5' for poster
#                self.vert_sbar = Scrollbar(canvases_frame, orient=VERTICAL, command=self.command_scroll_y)
#                canvas.config(yscrollcommand=self.vert_sbar.set)
#                self.vert_sbar.grid(row=0, column=i*2+1, sticky=N+S)
#                self.horiz_sbar = Scrollbar(canvases_frame, orient=HORIZONTAL, command=self.command_scroll_x)
#                canvas.config(xscrollcommand=self.horiz_sbar.set)
#                self.horiz_sbar.grid(row=1, column=i*2, sticky=E+W)

        # Canvas configuration - in pixels
        self.left_margin = 5
        self.right_margin = 5
        self.top_margin = 5
        self.bottom_margin = 5

        self.vert_axis_text_font_family = 'Courier'
        if poster:
            self.vert_axis_text_font_size = 30
        else:
            self.vert_axis_text_font_size = 15 #30 for poster
        self.vert_axis_text_font = '%s %d' % (self.vert_axis_text_font_family, self.vert_axis_text_font_size)
        self.vert_axis_text_gap_width = 5
        self.vert_axis_tick_length = 8
        self.vert_axis_tick_width = 3
        self.vert_axis_line_width = 3

        self.horiz_axis_line_width = 3
        self.horiz_axis_tick_length = 8
        self.horiz_axis_tick_width = 3
        self.horiz_axis_text_gap_height = 5
        self.horiz_axis_text_font_family = 'Courier'
        if poster:
            self.horiz_axis_text_font_size = 30 #30 for poster
        else:
            self.horiz_axis_text_font_size = 15 #30 for poster

        self.horiz_axis_text_font = '%s %d' % (self.horiz_axis_text_font_family, self.horiz_axis_text_font_size)

        self.vert_axis_char_width = int(self.vert_axis_text_font_size * 0.8125)
        self.vert_axis_char_height = int(self.vert_axis_text_font_size * 1.15)
        self.vert_axis_text_width = self.vert_axis_char_width * 8

        self.horiz_axis_char_width = int(self.horiz_axis_text_font_size * 0.8125)
        self.horiz_axis_char_height = int(self.horiz_axis_text_font_size * 1.15)
        self.horiz_axis_text_height = self.horiz_axis_char_height

        self.adj_top_margin = self.top_margin + self.vert_axis_char_height // 2
        self.adj_right_margin = self.right_margin + self.horiz_axis_char_width * 3

        # POSTER COLORS
        if poster:
            self.color_axis_label = 'black'
            self.color_axis_tick = 'black'
            self.color_axis_line = 'black'
            self.color_clump = '#B8372A'
            self.color_clump_highlighted = 'white'
            self.color_clump_selected_fill = '#707070'
            self.color_chain = '#207cc2'
            self.color_chain_highlighted = '#3e75ff'
            self.width_chain = 4
            self.width_chain_selected = 4
            self.color_anchor_chain = '#2a6cb6'
            self.color_chain_straight = '#23c2de'
            self.color_chain_straight_highlighted = '#0accb4'
            self.color_profile = 'cyan'
            self.color_wavelet = 'yellow'

        #MONITOR COLORS
        else:
            self.color_axis_label = 'yellow'
            self.color_axis_tick = 'white'
            self.color_axis_line = 'white'
            self.color_clump = '#c0c0c0'
            self.color_clump_highlighted = 'white'
            self.color_clump_selected_fill = '#707070'
            self.color_chain = '#c00000'
            self.color_chain_highlighted = '#ff0000'
            self.width_chain = 2
            self.width_chain_selected = 4
            self.color_anchor_chain = 'red'
            self.color_chain_straight = '#c08080'
            self.color_chain_straight_highlighted = '#ffc0c0'
            self.color_profile = 'cyan'
            self.color_wavelet = 'yellow'
#
        self.color_drag_rect = 'green'

        # Various other object parameters
        self.clump_y_halfwidth = 5
        self.chain_anchor_halfwidth = 3
        self.chain_highlight_sensitivity = 2

        # Initial zoom positions for longitude - zoom for ET is set with update_clump_db
        self.disp_global_long_min = 0.
        self.disp_global_long_max = 360.
        self.disp_long_min = self.disp_global_long_min
        self.disp_long_max = self.disp_global_long_max

        self.canvas_frame.pack(side=TOP)

        # Control sliders
        controls_parent_frame = Frame(self)
        control_frame = Frame(controls_parent_frame)

        gridrow = 0
        gridcol = 0

        self.var_clumpsize_min = DoubleVar()
        self.var_clumpsize_max = DoubleVar()
        self.var_clump_sigma_max = DoubleVar()
        self.var_clump_sigma_min = DoubleVar()

        label = Label(control_frame, text='Disp Clump Size Max')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.scale_clumpsize_max = Scale(control_frame, orient=HORIZONTAL, resolution=0.1,
                                         variable=self.var_clumpsize_max,
                                         command=self._command_refresh_clumps_chains_wavelets)
        self.scale_clumpsize_max.grid(row=gridrow, column=gridcol+1)
        gridrow += 1

        label = Label(control_frame, text='Disp Clump Size Min')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.scale_clumpsize_min = Scale(control_frame, orient=HORIZONTAL, resolution=0.1,
                                         variable=self.var_clumpsize_min,
                                         command=self._command_refresh_clumps_chains_wavelets)
        self.scale_clumpsize_min.grid(row=gridrow, column=gridcol+1)
        gridrow += 1

        label = Label(control_frame, text='Disp Clump Sigma Max')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.scale_clump_sigma_max = Scale(control_frame, orient=HORIZONTAL, resolution=0.1,
                                           variable=self.var_clump_sigma_max,
                                           command=self._command_refresh_clumps_chains_wavelets)
        self.scale_clump_sigma_max.grid(row=gridrow, column=gridcol+1)
        self.scale_clump_sigma_max.config(from_=0, to=20)
        self.var_clump_sigma_max.set(.05)
        gridrow += 1

        label = Label(control_frame, text='Disp Clump Sigma Min')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.scale_clump_sigma_min = Scale(control_frame, orient=HORIZONTAL, resolution=0.1,
                                           variable=self.var_clump_sigma_min,
                                           command=self._command_refresh_clumps_chains_wavelets)
        self.scale_clump_sigma_min.grid(row=gridrow, column=gridcol+1)
        self.scale_clump_sigma_min.config(from_=0, to=20)
        self.var_clump_sigma_min.set(.05)
        gridrow += 1



        self.var_draw_profiles = IntVar()
        self.var_draw_wavelets = IntVar()
        self.var_normalize_profiles = IntVar()
        self.var_log_profiles = IntVar()
        self.var_profile_scale = DoubleVar()
        self.var_profile_smoothing = DoubleVar()

        cbutton = Checkbutton(control_frame, text='Draw Profiles', variable=self.var_draw_profiles,
                              command=self._command_refresh_profiles)
        self.var_draw_profiles.set(0)
        cbutton.grid(row=gridrow, column=gridcol, sticky=W)

        cbutton = Checkbutton(control_frame, text='Normalize Profiles', variable=self.var_normalize_profiles,
                              command=self._command_refresh_profiles)
        self.var_normalize_profiles.set(0)
        cbutton.grid(row=gridrow, column=gridcol+1, sticky=W)
        gridrow += 1

        cbutton = Checkbutton(control_frame, text='Draw Wavelets', variable=self.var_draw_wavelets,
                              command=self._command_refresh_profiles)
        self.var_draw_wavelets.set(0)
        cbutton.grid(row=gridrow, column=gridcol, sticky=W)

        cbutton = Checkbutton(control_frame, text='Log Profiles', variable=self.var_log_profiles,
                              command=self._command_refresh_profiles)
        self.var_log_profiles.set(0)
        cbutton.grid(row=gridrow, column=gridcol+1, sticky=W)
        gridrow += 1

        label = Label(control_frame, text='Profile Scale')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.scale_profile = Scale(control_frame, orient=HORIZONTAL, resolution=1,
                                   variable=self.var_profile_scale,
                                   command=lambda x: self._command_refresh_profiles())
        self.scale_profile.config(from_=5, to=200)
        self.var_profile_scale.set(5)
        self.scale_profile.grid(row=gridrow, column=gridcol+1)
        gridrow += 1

        label = Label(control_frame, text='Profile Smoothing')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.scale_profile = Scale(control_frame, orient=HORIZONTAL, resolution=.1,
                                   variable=self.var_profile_smoothing,
                                   command=lambda x: self._command_refresh_profiles())
        self.scale_profile.config(from_=0., to=5.)
        self.var_profile_scale.set(1)
        self.scale_profile.grid(row=gridrow, column=gridcol+1)
        gridrow += 1

        info_frame = Frame(controls_parent_frame)

        gridrow = 0
        gridcol = 0

        label = Label(info_frame, text='Co-rot Longitude:')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.label_longitude = Label(info_frame, text='')
        self.label_longitude.grid(row=gridrow, column=gridcol+1, sticky=W)
        gridrow += 1

        label = Label(info_frame, text='Time:')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.label_time = Label(info_frame, text='')
        self.label_time.grid(row=gridrow, column=gridcol+1, sticky=W)
        gridrow += 1

        label = Label(info_frame, text='Closest OBSID:')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.label_obsid = Label(info_frame, text='')
        self.label_obsid.grid(row=gridrow, column=gridcol+1, sticky=W)
        gridrow += 1

        label = Label(info_frame, text='Inertial Longitude:')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.label_inertial_longitude = Label(info_frame, text='')
        self.label_inertial_longitude.grid(row=gridrow, column=gridcol+1, sticky=W)
        gridrow += 1

        label = Label(info_frame, text='Fake True Anomaly:')
        label.grid(row=gridrow, column=gridcol, sticky=W)
        self.label_true_anomaly = Label(info_frame, text='')
        self.label_true_anomaly.grid(row=gridrow, column=gridcol+1, sticky=W)
        gridrow += 1

        self.label_clumps = []
        for clump_num in range(1,5):
            label = Label(info_frame, text='Clump %d:'%clump_num)
            label.grid(row=gridrow, column=gridcol, sticky=W)
            label_clump = Label(info_frame, text='')
            label_clump.grid(row=gridrow, column=gridcol+1, sticky=W)
            self.label_clumps.append(label_clump)
            gridrow += 1

        self.label_chains= []
        for chain_num in range(1,5):
            label = Label(info_frame, text='Chain %d:'%chain_num)
            label.grid(row=gridrow, column=gridcol, sticky=W)
            label_chain = Label(info_frame, text='')
            label_chain.grid(row=gridrow, column=gridcol+1, sticky=W)
            self.label_chains.append(label_chain)
            gridrow += 1

        control_frame.grid(row=0, column=0, sticky=NW)
        info_frame.grid(row=0, column=1, sticky=NW)

        controls_parent_frame.pack(side=LEFT)
        self.pack()

        self.left_mousebutton_down = False # Select or drag-z
        self.left_mousebutton_down_pos = None
        self.left_mousebutton_down_pix = None
        self.drag_rect_obj = None
        self.right_mousebutton_down = False # Pan
        self.active_panning = False

        # Have to do this before binding events
        self.update_clump_db(clump_db, redraw=False)
        self.update_chain_list(chain_list, redraw=False)
        self.refresh_display()

        # Bind events
        self.canvas.bind("<Button-1>", lambda event: self._b1press_callback_handler(event))
        self.canvas.bind("<ButtonRelease-1>", lambda event: self._b1release_callback_handler(event, False, False))
        self.canvas.bind("<Shift-ButtonRelease-1>", lambda event: self._b1release_callback_handler(event, True, False))
        self.canvas.bind("<Control-ButtonRelease-1>", lambda event: self._b1release_callback_handler(event, False, True))
        self.canvas.bind("<Shift-Control-ButtonRelease-1>", lambda event: self._b1release_callback_handler(event, True, True))
        self.canvas.bind("<Button-3>", lambda event: self._b3press_callback_handler(event))
        self.canvas.bind("<ButtonRelease-3>", lambda event: self._b3release_callback_handler(event))
        self.canvas.bind("<Motion>", lambda event: self._mousemove_callback_handler(event))

    def update_clump_db(self, clump_db, redraw=True):
        """Update the clumps with a brand new clump DB"""
        self.clump_db = clump_db
        self.all_clumps_list = []
        for obsid in list(clump_db.keys()):
            clump_db_entry = clump_db[obsid]
            for clump in clump_db_entry.clump_list:
                clump.clump_db_entry = clump_db_entry
                clump.canvas_obj = None
                clump.highlighted = False
                clump.selected = False
                self.all_clumps_list.append(clump)

        et_min = 1.e38
        et_max = 0.
        for obsid, clump_db_entry in list(self.clump_db.items()):
            et_min = min(et_min, clump_db_entry.et_min)
            et_max = max(et_max, clump_db_entry.et_max)

        # Add a little slop so objects aren't right up against the margins
        self.disp_global_et_min = et_min - (et_max-et_min)*.05
        self.disp_global_et_max = et_max + (et_max-et_min)*.05

        self.disp_et_min = self.disp_global_et_min
        self.disp_et_max = self.disp_global_et_max

        # Figure out the min and max clumpsize
        self.global_clumpsize_min = 1e38
        self.global_clumpsize_max = 0.
        self.global_clump_sigma_max = 0.
        self.global_clump_sigma_min = 1e38

        for obsid, clump_db_entry in list(self.clump_db.items()):
            for clump in clump_db_entry.clump_list:
                self.global_clumpsize_min = min(self.global_clumpsize_min, clump.scale)
                self.global_clumpsize_max = max(self.global_clumpsize_max, clump.scale)
                self.global_clump_sigma_min = min(self.global_clump_sigma_min, clump.fit_sigma)
                self.global_clump_sigma_max = max(self.global_clump_sigma_max, clump.fit_sigma)

#        self.scale_clumpsize_min.config(from_=self.global_clumpsize_min, to=self.global_clumpsize_max)
        self.var_clumpsize_min.set(self.global_clumpsize_min)
        self.scale_clumpsize_max.config(from_=self.global_clumpsize_min, to=self.global_clumpsize_max)
        self.var_clumpsize_max.set(self.global_clumpsize_max)

        self.var_clump_sigma_min.set(self.global_clump_sigma_min)
        self.scale_clump_sigma_max.config(from_=self.global_clump_sigma_min, to=self.global_clump_sigma_max)
        self.var_clump_sigma_max.set(self.global_clump_sigma_max)

        if redraw:
            self.refresh_display()

    def update_chain_list(self, chain_list, redraw=True):
        """Update the clump chains with a brand new clump chain list"""
        self.all_chains_list = chain_list
        for chain in chain_list:
            chain.highlighted = False
            try:
                foo = chain.selected
            except:
                # Keep the selected flag if it was already set
                chain.selected = False
            try:
                foo = chain.base_long
            except:
                chain.base_long = chain.clump_list[0].g_center
            chain.chain_link_obj_list = None
        if redraw:
            self.refresh_display()

    def selected_clumps(self):
        """Return a list of selected clumps."""
        sel_clumps = [x for x in self.all_clumps_list if x.selected]
        return sel_clumps

    def selected_chains(self):
        """Return a list of selected chains."""
        sel_chains = [x for x in self.all_chains_list if x.selected]
        return sel_chains

    def displayed_clumps(self):
        """Return a list of clumps currently displayed."""
        disp_clumps = [x for x in self.all_clumps_list if x.canvas_obj is not None]
        return disp_clumps

    def _update_canvas_dimensions(self):
        self.drawing_area_x_start = (self.left_margin + self.vert_axis_text_width +
                                     self.vert_axis_text_gap_width + self.vert_axis_tick_length +
                                     self.vert_axis_line_width)
        self.drawing_area_x_end = self.canvas_size_x - self.adj_right_margin - 1
        self.drawing_area_width = self.drawing_area_x_end - self.drawing_area_x_start + 1

        self.drawing_area_y_start = self.adj_top_margin
        self.drawing_area_y_end = (self.canvas_size_y - self.bottom_margin -
                                   self.horiz_axis_text_gap_height - self.horiz_axis_text_height -
                                   self.horiz_axis_tick_length -
                                   self.horiz_axis_line_width - 1)
        self.drawing_area_height = self.drawing_area_y_end - self.drawing_area_y_start + 1

        self.vert_axis_num_ticks = self.drawing_area_height // self.vert_axis_char_height // 3
        self.horiz_axis_num_ticks = self.drawing_area_width // self.horiz_axis_char_width // (6*2)

    def _draw_axes(self):
        self.canvas.delete('axislines')
        # Vert axis
        y_axis_x_ctr = self.drawing_area_x_start - self.vert_axis_line_width//2 - 1
        y_axis_y_end = self.drawing_area_y_end + self.horiz_axis_line_width
        self.canvas.create_line(y_axis_x_ctr, self.adj_top_margin, y_axis_x_ctr, y_axis_y_end,
                                fill=self.color_axis_line,
                                width=self.vert_axis_line_width, tags='axislines')
        # Horiz axis
        x_axis_y_ctr = self.drawing_area_y_end + self.horiz_axis_line_width//2 + 1
        x_axis_x_start = self.drawing_area_x_start - self.vert_axis_line_width
        self.canvas.create_line(x_axis_x_start, x_axis_y_ctr,
                                self.drawing_area_x_end + self.horiz_axis_tick_width//2, x_axis_y_ctr,
                                fill=self.color_axis_line,
                                width=self.horiz_axis_line_width, tags='axislines')

    def _draw_axis_labels(self):
        self.canvas.delete('axis')
        # Draw vert axis labels and ticks
        tick_x_start = self.drawing_area_x_start - (self.vert_axis_line_width//2)*2 - self.vert_axis_tick_length - 1
        tick_x_end = self.drawing_area_x_start - (self.vert_axis_line_width//2)*2 - 1
        for num in range(self.vert_axis_num_ticks):
            y_ctr = int(float(self.drawing_area_height - self.vert_axis_tick_width // 2) /
                        (self.vert_axis_num_ticks-1) * num +
                        self.adj_top_margin + self.vert_axis_tick_width // 2)
            self.canvas.create_line(tick_x_start, y_ctr, tick_x_end, y_ctr,
                                    fill=self.color_axis_tick,
                                    width=self.vert_axis_tick_width,
                                    tags='axis')
            time_text = clump_util.et2utc(float(self.disp_et_max-self.disp_et_min) /
                                      (self.vert_axis_num_ticks-1) * (self.vert_axis_num_ticks-num-1) +
                                      self.disp_et_min,
                                      'D', 0)
            time_text = time_text[:8]
            self.canvas.create_text(self.left_margin, y_ctr,
                                    anchor='w', font=self.vert_axis_text_font,
                                    text=time_text, fill=self.color_axis_label, tags='axis')

        # Draw horiz axis labels and ticks
        tick_y_start = self.drawing_area_y_end + self.horiz_axis_line_width + 1
        tick_y_end = self.drawing_area_y_end + self.horiz_axis_line_width + self.horiz_axis_tick_length
        for num in range(self.horiz_axis_num_ticks):
            x_ctr = int(float(self.drawing_area_width) / (self.horiz_axis_num_ticks-1) * num +
                        self.drawing_area_x_start - self.horiz_axis_tick_width // 2)
            self.canvas.create_line(x_ctr, tick_y_start, x_ctr, tick_y_end,
                                    fill=self.color_axis_tick,
                                    width=self.horiz_axis_tick_width,
                                    tags='axis')
            longitude = ((self.disp_long_max-self.disp_long_min) / (self.horiz_axis_num_ticks-1) * num +
                         self.disp_long_min)
            long_text = '%.2f' % longitude
            self.canvas.create_text(x_ctr, self.canvas_size_y-self.bottom_margin,
                                    anchor='s', font=self.horiz_axis_text_font,
                                    text=long_text, fill=self.color_axis_label,
                                    tags='axis')

    def _long_to_x(self, long):
        x_ctr = int(self.drawing_area_x_start +
                    (int-self.disp_long_min) / (self.disp_long_max-self.disp_long_min) *
                    self.drawing_area_width)
        return x_ctr

    def _et_to_y(self, et):
        y_ctr = int(self.drawing_area_y_start +
                    self.drawing_area_height -
                    ((et-self.disp_et_min) / (self.disp_et_max-self.disp_et_min) *
                     self.drawing_area_height))
        return y_ctr

    def _clump_xy(self, clump):
        return self._long_to_x(clump.g_center), self._et_to_y(clump.clump_db_entry.et)

    def _draw_clumps(self):
        self.canvas.delete('clump')

        clumpsize_min = self.var_clumpsize_min.get()
        clumpsize_max = self.var_clumpsize_max.get()
        clump_sigma_max = self.var_clump_sigma_max.get()
        clump_sigma_min = self.var_clump_sigma_min.get()

        for clump in self.all_clumps_list:
            clump.canvas_obj = None
            # Is the ET in the current zoom range?
            if clump.clump_db_entry.et < self.disp_et_min or clump.clump_db_entry.et > self.disp_et_max:
                continue
            # Is the clump size in the currently selected range?
            if clump.fit_width_deg < clumpsize_min or clump.fit_width_deg > clumpsize_max:
                continue
            # Is the longitude in the current zoom range?
            if clump.g_center < self.disp_long_min or clump.g_center > self.disp_long_max:
                continue
            #Is the clump within specified sigma range?
            if clump.fit_sigma < clump_sigma_min or clump.fit_sigma > clump_sigma_max:
                continue

            x_ctr, y_ctr = self._clump_xy(clump)
            x_halfwidth = int(clump.scale/2 / (self.disp_long_max-self.disp_long_min) *
                              self.drawing_area_width)
#            if (x_ctr-x_halfwidth < self.drawing_area_x_start or
#                x_ctr+x_halfwidth > self.drawing_area_x_end or
#                y_ctr-self.clump_y_halfwidth < self.drawing_area_y_start or
#                y_ctr+self.clump_y_halfwidth > self.drawing_area_y_end):
#                continue
            if (y_ctr-self.clump_y_halfwidth < self.drawing_area_y_start or
                y_ctr+self.clump_y_halfwidth > self.drawing_area_y_end):
                continue
            fill_color = ''
            #for poster - add width = 2
            if poster:
                clump_width = 2
            else:
                clump_width = 1.
            clump.canvas_obj = self.canvas.create_oval(x_ctr-x_halfwidth, y_ctr-self.clump_y_halfwidth,
                                                       x_ctr+x_halfwidth, y_ctr+self.clump_y_halfwidth,
                                                       outline=self.color_clump, fill=fill_color, tags='clump', width = clump_width)
            self._update_clump_selection(clump)

    def _update_clump_selection(self, clump):
        if not clump.canvas_obj:
            return
        if clump.selected:
            fill_color = self.color_clump_selected_fill
        else:
            fill_color = ''
        self.canvas.itemconfig(clump.canvas_obj, fill=fill_color)
        if clump.selected:
            self.canvas.tag_lower(clump.canvas_obj) # Put on the bottom so other clumps draw on top

    def _update_clump_highlight(self, clump):
        if clump.highlighted:
            color = self.color_clump_highlighted
        else:
            color = self.color_clump
        self.canvas.itemconfig(clump.canvas_obj, outline=color)

    def _update_chain_selection(self, chain):
        if chain.selected:
            width = self.width_chain_selected
        else:
            width = self.width_chain
        if chain.chain_link_obj_list:
            for link_obj in chain.chain_link_obj_list:
                self.canvas.itemconfig(link_obj, width=width)
                if chain.selected:
                    self.canvas.tag_raise(link_obj) # Put on the top

    def _update_chain_highlight(self, chain):
        if chain.highlighted:
            fill_color = self.color_chain_highlighted
            straight_color = self.color_chain_straight_highlighted
        else:
            fill_color = self.color_chain
            straight_color = self.color_chain_straight
        if chain.chain_link_obj_list:
            for link_obj in chain.chain_link_obj_list:
                self.canvas.itemconfig(link_obj, fill=fill_color)
                if chain.highlighted:
                    self.canvas.tag_raise(link_obj) # Put on the top
            self.canvas.itemconfig(chain.straight_obj, fill=straight_color) #comment for poster

    def _draw_chains(self):
        self.canvas.delete('chain')

        clumpsize_min = self.var_clumpsize_min.get()
        clumpsize_max = self.var_clumpsize_max.get()
        clump_sigma_max = self.var_clump_sigma_max.get()
        clump_sigma_min = self.var_clump_sigma_min.get()

        for chain_entry in self.all_chains_list:
            chain_entry.chain_link_obj_list = None
            x_list = []
            y_list = []
            et_list = []
            bad_chain = False
            for clump in chain_entry.clump_list:
#
                if clump.fit_width_deg < clumpsize_min or clump.fit_width_deg > clumpsize_max:
                    bad_chain = True
                    break
#
                if clump.fit_sigma < clump_sigma_min or clump.fit_sigma > clump_sigma_max:
                    bad_chain = True
                    break
                x_ctr, y_ctr = self._clump_xy(clump)

#                    x_ctr < self.drawing_area_x_start or
#                    x_ctr > self.drawing_area_x_end or
                if (y_ctr < self.drawing_area_y_start or
                    y_ctr > self.drawing_area_y_end):
                    bad_chain = True
                    break
                x_list.append(x_ctr)
                y_list.append(y_ctr)
                et_list.append(clump.clump_db_entry.et)
            if not bad_chain:
                chain_entry.chain_link_obj_list = []
                for n in range(len(x_list)-1):
                    chain_link_entry = self.canvas.create_line(x_list[n], y_list[n], x_list[n+1], y_list[n+1],
                                                               fill=self.color_chain, tags='chain')
                    chain_entry.chain_link_obj_list.append(chain_link_entry)
                self._update_chain_selection(chain_entry)
                for n in range(len(x_list)):
                    self.canvas.create_oval(x_list[n]-self.chain_anchor_halfwidth, y_list[n]-self.chain_anchor_halfwidth,
                                            x_list[n]+self.chain_anchor_halfwidth, y_list[n]+self.chain_anchor_halfwidth,
                                            fill=self.color_anchor_chain, tags='chain')
                # Draw the straight line - comment out for poster
                if poster == False:
                    straight_obj = self.canvas.create_line(self._long_to_x(chain_entry.base_long), y_list[0],
                                                           self._long_to_x(chain_entry.base_long+
                                                                           (et_list[-1]-et_list[0])*chain_entry.rate),
                                                           y_list[-1], fill=self.color_chain_straight, tags='chain')
                    chain_entry.straight_obj = straight_obj
#
    def _plot_a_clump(self, sub_long, mexhat, long_step, y_base):
        # Internal helper for _draw_profiles
        for idx in range(0, len(sub_long)-long_step-1, long_step):
            # We quantize the index so that small shifts in longitude due to panning don't cause
            # visual jumps in the profile display as different indices are sampled
            quant_idx = (idx // long_step) * long_step
            if (sub_long[quant_idx] < self.disp_long_min or
                sub_long[quant_idx+long_step] > self.disp_long_max):
                continue
            x0 = self._long_to_x(sub_long[quant_idx])
            x1 = self._long_to_x(sub_long[quant_idx+long_step])
            y0 = y_base-mexhat[quant_idx]
            y1 = y_base-mexhat[quant_idx+long_step]
            self.canvas.create_line(x0, y0, x1, y1,
                                    fill=self.color_wavelet, tags='wavelet')

    def _draw_profiles(self, draw_profiles=True, draw_wavelets=True):
        if draw_profiles:
            self.canvas.delete('profile')
        if draw_wavelets:
            self.canvas.delete('wavelet')

        if self.var_draw_profiles.get() == 0:
            draw_profiles = False
        if self.var_draw_wavelets.get() == 0:
            draw_wavelets = False

        if not draw_profiles:
            # If we don't draw profiles, don't draw wavelets either
            return

        normalize_profiles = self.var_normalize_profiles.get()
        profile_log = self.var_log_profiles.get()
        profile_scale = self.var_profile_scale.get()
        profile_smooth = self.var_profile_smoothing.get()

        for obsid, clump_db_entry in list(self.clump_db.items()):
            # Is the ET in the current zoom range?
            if clump_db_entry.et < self.disp_et_min or clump_db_entry.et > self.disp_et_max:
                continue
            ew_data = clump_db_entry.ew_data
            long_res = 360. / len(clump_db_entry.ew_data)
            long_start_idx = max(int(self.disp_long_min / long_res), 0)
            long_end_idx = min(int(self.disp_long_max / long_res), len(ew_data)-1)
            if profile_smooth != 0.:
                profile_smooth2 = profile_smooth // long_res // 2
                ew_data_new = ma.zeros(ew_data.shape[0])
                for n in range(long_start_idx, long_end_idx+1):
                    if ew_data.mask[n]:
                        ew_data_new[n] = ma.masked
                    else:
                        ew_data_new[n] = ma.mean(ew_data[max(n-profile_smooth2,0):
                                                         min(n+profile_smooth2+1,len(ew_data)-1)])
                ew_data = ew_data_new
            if profile_log:
                ew_data = np.log(np.maximum(ew_data, 1e-6))
            profile_min = ma.min(ew_data[long_start_idx:long_end_idx+1])
            ew_data = ew_data-profile_min
            normalize_scale = 1.
            if normalize_profiles:
                normalize_scale = 1. / ma.max(ew_data[long_start_idx:long_end_idx+1])
            ew_data = ew_data*normalize_scale*profile_scale

            y_base = self._et_to_y(clump_db_entry.et) + profile_scale/2
            long_step = max(1, int(long_end_idx-long_start_idx)/500)

            if draw_profiles:
                for idx in range(long_start_idx, long_end_idx-long_step+1, long_step): # Intentionally don't include last idx
                    # We quantize the index so that small shifts in longitude due to panning don't cause
                    # visual jumps in the profile display as different indices are sampled
                    quant_idx = (idx // long_step) * long_step
                    ew_data_mask = ma.getmaskarray(ew_data)
                    if not ew_data_mask[quant_idx] and not ew_data_mask[quant_idx+long_step]:
                        x0 = self._long_to_x(quant_idx*long_res)
                        x1 = self._long_to_x((quant_idx+long_step)*long_res)
                        y0 = y_base-ew_data[quant_idx]
                        y1 = y_base-ew_data[quant_idx+long_step]
                        self.canvas.create_line(x0, y0, x1, y1,
                                                fill=self.color_profile, tags='profile')

            if draw_wavelets:
                for clump in clump_db_entry.clump_list:
                    if not (self.disp_long_min-clump.scale < clump.longitude < self.disp_long_max+clump.scale):
                        continue
#
                    if not (self.var_clumpsize_min.get() < clump.fit_width_deg < self.var_clumpsize_max.get()):
                        continue
#
                    if not (self.var_clump_sigma_min.get() < clump.fit_sigma < self.var_clump_sigma_max.get()):
                        continue
                    longitudes = np.arange(len(ew_data)*3) * long_res - 360.

                    clump_disp_scale = 0.5

                    mother_wavelet = cwt.SDG(len_signal=len(ew_data)*3, scales=np.array([int(clump.scale_idx/2)]))
                    mexhat = mother_wavelet.coefs[0].real # Get the master wavelet shape
                    mh_start_idx = round(len(mexhat)/2.-clump.scale_idx*clump_disp_scale)
                    mh_end_idx = round(len(mexhat)/2.+clump.scale_idx*clump_disp_scale)
                    mexhat = mexhat[mh_start_idx:mh_end_idx+1] # Extract just the positive part
                    mexhat = mexhat*clump.mexhat_height+clump.mexhat_base
                    if profile_log:
                        mexhat = np.log(np.maximum(mexhat, 1e-6))
                    mexhat = mexhat-profile_min
                    mexhat = mexhat*normalize_scale*profile_scale
                    longitude_idx = clump.longitude_idx
                    if longitude_idx+clump.scale_idx*clump_disp_scale >= len(ew_data):
                        # Runs off right side - make it run off left side instead
                        longitude_idx -= len(ew_data)
                    sub_long = longitudes[longitude_idx-clump.scale_idx*clump_disp_scale+len(ew_data):
                                          longitude_idx-clump.scale_idx*clump_disp_scale+len(mexhat)+len(ew_data)] # Longitude range in data
                    self._plot_a_clump(sub_long, mexhat, long_step, y_base)
                    if longitude_idx-clump.scale_idx*clump_disp_scale < 0: # Runs off left side - plot it twice
                        self._plot_a_clump(sub_long+360., mexhat, long_step, y_base)


    def _win_xy_to_long_et(self, x, y):
        et = ((self.drawing_area_height - (y - self.drawing_area_y_start)) / float(self.drawing_area_height) *
              (self.disp_et_max-self.disp_et_min)) + self.disp_et_min
        long = ((x - self.drawing_area_x_start) / float(self.drawing_area_width) *
                (self.disp_long_max-self.disp_long_min)) + self.disp_long_min
        if (int < self.disp_long_min or int > self.disp_long_max or
            et < self.disp_et_min or et > self.disp_et_max):
            return None, None
        return int, et

    def _win_clip_to_drawing_area(self, x, y):
        x = max(x, self.drawing_area_x_start)
        x = min(x, self.drawing_area_x_end)
        y = max(y, self.drawing_area_y_start)
        y = min(y, self.drawing_area_y_end)
        return x, y

    def _draw_clumps_chains(self):
        # The order here is important so the proper objects are "underneath" others
        self._draw_clumps()
        self._draw_chains()

    def refresh_display(self):
        # The order here is important so the proper objects are "underneath" others
        self._update_canvas_dimensions()
        self._draw_axes()
        self._draw_axis_labels()
        if not self.active_panning:
            self._draw_profiles()
        self._draw_clumps_chains()

    def _command_refresh_clumps_chains(self, val):
        self._draw_clumps_chains()

    def _command_refresh_clumps_chains_wavelets(self, val):
        self._draw_clumps_chains()
        self._draw_profiles(draw_profiles=False)

    def _command_refresh_profiles(self):
        self._draw_profiles()

    def _clumps_under_mouse(self, x, y):
        longitude, et = self._win_xy_to_long_et(x, y)
        in_range_clumps = []
        for clump in self.all_clumps_list:
            temp, max_et = self._win_xy_to_long_et(x, y-self.clump_y_halfwidth)
            temp, min_et = self._win_xy_to_long_et(x, y+self.clump_y_halfwidth)
            if min_et > clump.clump_db_entry.et or max_et < clump.clump_db_entry.et:
                continue # Not in the vertical extent of the clump oval
            if clump.g_center-clump.scale/2 > longitude or clump.g_center+clump.scale/2 < longitude:
                continue # Not in the horizontal extent of the clump oval
            in_range_clumps.append(clump)
        return in_range_clumps

    def _chains_under_mouse(self, x, y):
        selected = self.canvas.find_overlapping(x-self.chain_highlight_sensitivity,
                                                y-self.chain_highlight_sensitivity,
                                                x+self.chain_highlight_sensitivity,
                                                y+self.chain_highlight_sensitivity)
        in_range_chains = []
        for chain in self.all_chains_list:
            if chain.chain_link_obj_list:
                for obj in chain.chain_link_obj_list:
                    if obj in selected:
                        in_range_chains.append(chain)
                        break
        return in_range_chains

    def _b1press_callback_handler(self, event):
        """Internal - callback for button-one press."""

        if self.left_mousebutton_down:
            return # How did we even get here?

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        x, y = self._win_clip_to_drawing_area(x, y)
        self.left_mousebutton_down = True
        self.left_mousebutton_down_pix = (x, y)
        self.left_mousebutton_down_pos = self._win_xy_to_long_et(x, y)
        self.drag_rect_obj = None

    def _b1release_callback_handler(self, event, shift_key, ctrl_key):
        """Internal - callback for button-one release."""

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        x, y = self._win_clip_to_drawing_area(x, y)
        if not self.left_mousebutton_down:
            return # How did we even get here?

        self.left_mousebutton_down = False
        if self.drag_rect_obj:
            self.canvas.delete(self.drag_rect_obj)
            self.drag_rect_obj = None
            if shift_key:
                # We were doing drag-select
                if not ctrl_key:
                    # First deselect everything if shift not held down
                    for clump in self.all_clumps_list:
                        if clump.selected:
                            clump.selected = False
                            self._update_clump_selection(clump)
                long_min = min(self.left_mousebutton_down_pos[0], self.drag_rect_end_pos[0])
                long_max = max(self.left_mousebutton_down_pos[0], self.drag_rect_end_pos[0])
                et_min = min(self.left_mousebutton_down_pos[1], self.drag_rect_end_pos[1])
                et_max = max(self.left_mousebutton_down_pos[1], self.drag_rect_end_pos[1])
                for clump in self.all_clumps_list:
                    if (clump.g_center-clump.scale/2 > long_min and
                        clump.g_center+clump.scale/2 < long_max and
                        clump.clump_db_entry.et > et_min and
                        clump.clump_db_entry.et < et_max):
                        clump.selected = True
                        self._update_clump_selection(clump)
            else:
                # We were doing drag-zoom
                self.disp_long_min = min(self.left_mousebutton_down_pos[0],
                                         self.drag_rect_end_pos[0])
                self.disp_long_max = max(self.left_mousebutton_down_pos[0],
                                         self.drag_rect_end_pos[0])
                self.disp_et_min = min(self.left_mousebutton_down_pos[1],
                                       self.drag_rect_end_pos[1])
                self.disp_et_max = max(self.left_mousebutton_down_pos[1],
                                       self.drag_rect_end_pos[1])
                self.refresh_display()
        else:
            # We weren't dragging, so this is an object select
            if not ctrl_key:
                # First deselect everything if shift not held down
                for clump in self.all_clumps_list:
                    if clump.selected:
                        clump.selected = False
                        self._update_clump_selection(clump)
                for chain in self.all_chains_list:
                    if chain.selected:
                        chain.selected = False
                        self._update_chain_selection(chain)
            # Find clumps under the mouse and toggle their selection
            in_range_clumps = self._clumps_under_mouse(x, y)
            for clump in in_range_clumps:
                if clump.canvas_obj:
                    clump.selected = not clump.selected
                    self._update_clump_selection(clump)
            # Find chains under the mouse and toggle their selection
            in_range_chains = self._chains_under_mouse(event.x, event.y)
            for chain in in_range_chains:
                if chain.chain_link_obj_list:
                    chain.selected = not chain.selected
                    self._update_chain_selection(chain)

    def _b3press_callback_handler(self, event):
        """Internal - callback for button-three press."""

        if self.right_mousebutton_down:
            return # How did we even get here?

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        x, y = self._win_clip_to_drawing_area(x, y)
        self.right_mousebutton_down = True
        self.right_mousebutton_down_pix = (x, y)
        self.right_mousebutton_down_pos = self._win_xy_to_long_et(x, y)

    def _b3release_callback_handler(self, event):
        """Internal - callback for button-three release."""

        if not self.right_mousebutton_down:
            return # How did we even get here?

        self.right_mousebutton_down = False

        if self.active_panning:
            # We're done with panning
            self.active_panning = False
            if self.var_draw_profiles.get() == 1:
                # We haven't been drawing profiles, so draw them now
                self.refresh_display()
        else:
            # It was just a click and unclick in the same spot - unzoom
            et_dist = self.disp_et_max - self.disp_et_min
            self.disp_et_max = min(self.disp_et_max+et_dist, self.disp_global_et_max)
            self.disp_et_min = max(self.disp_et_min-et_dist, self.disp_global_et_min)
            long_dist = self.disp_long_max - self.disp_long_min
            self.disp_long_max = min(self.disp_long_max+long_dist, self.disp_global_long_max)
            self.disp_long_min = max(self.disp_long_min-long_dist, self.disp_global_long_min)
            self.refresh_display()

    def _mousemove_callback_handler(self, event):
        """Internal - callback for mouse move."""

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        x, y = self._win_clip_to_drawing_area(x, y)
        longitude, et = self._win_xy_to_long_et(x, y)

        # Deal with zooming and panning

        if self.left_mousebutton_down:
            if not self.drag_rect_obj and not self.right_mousebutton_down:
                # Decide if we can start the drag rect
                if np.sqrt((x-self.left_mousebutton_down_pix[0])**2+
                           (y-self.left_mousebutton_down_pix[1])**2) > 4:
                    # We've dragged enough - let the zooming start
                    self.drag_rect_obj = self.canvas.create_rectangle(self.left_mousebutton_down_pix[0],
                                                                      self.left_mousebutton_down_pix[1],
                                                                      x, y, fill='',
                                                                      outline=self.color_drag_rect)
            if self.drag_rect_obj:
                self.canvas.coords(self.drag_rect_obj,
                                   self.left_mousebutton_down_pix[0],
                                   self.left_mousebutton_down_pix[1],
                                   x, y)
                self.drag_rect_end_pix = (x, y)
                self.drag_rect_end_pos = (longitude, et)

        if self.right_mousebutton_down:
            if not self.active_panning and not self.left_mousebutton_down:
                # Decide if we can start panning
                if np.sqrt((x-self.right_mousebutton_down_pix[0])**2+
                           (y-self.right_mousebutton_down_pix[1])**2) > 4:
                    # We've dragged enough - let the panning start
                    self.active_panning = True
                    self.active_panning_start = (self.disp_long_min, self.disp_long_max,
                                                 self.disp_et_min, self.disp_et_max)
                    self.canvas.delete('profile')
                    self.canvas.delete('wavelet')
            if self.active_panning:
                delta_long = (float(self.right_mousebutton_down_pix[0] - x) / self.drawing_area_width *
                              (self.disp_long_max-self.disp_long_min))
                delta_et = (float(y - self.right_mousebutton_down_pix[1]) / self.drawing_area_height *
                            (self.disp_et_max-self.disp_et_min))
                self.disp_long_min = self.active_panning_start[0] + delta_long
                self.disp_long_max = self.active_panning_start[1] + delta_long
                self.disp_et_min = self.active_panning_start[2] + delta_et
                self.disp_et_max = self.active_panning_start[3] + delta_et
                self.refresh_display()

        # Update the info pane

        if longitude is None or et is None:
            self.label_longitude.config(text='')
            self.label_time.config(text='')
            self.label_obsid.config(text='')
            self.label_inertial_longitude.config(text='')
            self.label_true_anomaly.config(text='')
        else:
            self.label_longitude.config(text='%6.2f' % longitude)
            self.label_time.config(text=clump_util.et2utc(et, 'C', 0))

            # Find closest obsid
            best_et_dist = 1e38
            best_obsid = None
            for obsid, clump_db_entry in list(self.clump_db.items()):
                if abs(et-clump_db_entry.et) < best_et_dist:
                    best_et_dist = abs(et-clump_db_entry.et)
                    best_obsid = obsid
                    best_et = clump_db_entry.et

            self.label_obsid.config(text=best_obsid+' ('+clump_util.et2utc(best_et, 'C', 0)+')')
            self.label_inertial_longitude.config(text='%6.2f' %
                                                 ((longitude-ringutil.ComputeLongitudeShift(best_et))%360.))
            self.label_true_anomaly.config(text='%6.2f' %
                                           ((longitude-ringutil.ComputeLongitudeShift(best_et)-2.7007*(best_et/8600))%360.))

            # Find clumps under the mouse
            in_range_clumps = self._clumps_under_mouse(x, y)
            for n in range(len(self.label_clumps)):
                self.label_clumps[n].config(text='')
            for n in range(min(len(in_range_clumps), len(self.label_clumps))):
                clump = in_range_clumps[n]
                str = 'Long %6.2f Scale %6.2f [%-17s %s]' % (clump.g_center, clump.fit_width_deg,
                                                             clump.clump_db_entry.obsid,
                                                             clump_util.et2utc(clump_db_entry.et, 'C', 0))
                self.label_clumps[n].config(text=str)

            # Unhighlight previously highlighted clumps
            for clump in self.all_clumps_list:
                if clump.canvas_obj and clump.highlighted:
                    clump.highlighted = False
                    self._update_clump_highlight(clump)
            # Highlight the clumps
            for clump in in_range_clumps:
                if clump.canvas_obj:
                    clump.highlighted = True
                    self._update_clump_highlight(clump)

            # Find clump chains under the mouse
            in_range_chains = self._chains_under_mouse(event.x, event.y)
            for n in range(len(self.label_chains)):
                self.label_chains[n].config(text='')
            for n in range(min(len(in_range_chains), len(self.label_chains))):
                chain = in_range_chains[n]
                str = 'Rate %.5f BaseLong %6.2f' % (chain.rate*86400, chain.base_long)
                for i, clump in enumerate(chain.clump_list):
                    str += ' / %6.2f (%6.2f)' % (clump.g_center, chain.long_err_list[i])
                self.label_chains[n].config(text=str)

            # Unhighlight previously highlighted clumps
            for chain in self.all_chains_list:
                if chain.chain_link_obj_list and chain.highlighted:
                    chain.highlighted = False
                    self._update_chain_highlight(chain)
            # Highlight the clumps
            for chain in in_range_chains:
                if chain.chain_link_obj_list:
                    chain.highlighted = True
                    self._update_chain_highlight(chain)
