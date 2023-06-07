###############################################################################
# imgdisp.py
#
# ImageDisp is a generic Tcl/Tk-based GUI interface for displaying images and
# associated metadata.
###############################################################################

from collections.abc import Sequence

import numpy as np
import scipy.ndimage.interpolation as ndinterp

from PIL import Image, ImageTk
if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
import tkinter as tk

#===============================================================================
#
# The ImageDisp Class
#
#===============================================================================

class ImageDisp(tk.Frame):
    """Display greyscale images with color overlays and image controls.

    A class that displays one or more greyscale images, with optional color
    overlays, and provides a variety of controls for image manipulation
    (e.g. white/blackpoint, gamma) allowing callbacks to be registered for
    various events."""

    def __init__(self,
                 imgdata_list,
                 overlay_list=None, color_column_list=None,
                 parent=None,
                 title=None,
                 canvas_size=None,
                 flip_y=False, origin=(0,0),
                 enlarge_limit=None, shrink_limit=None, one_zoom=True,
                 auto_update=True,
                 overlay_transparency=0.55,
                 separate_contrast=False,
                 blackpoint=None, blackpoint_min=None, blackpoint_max=None,
                 whitepoint=None, whitepoint_min=None, whitepoint_max=None,
                 whitepoint_ignore_frac=1.00,
                 gamma=0.5, gamma_min=0., gamma_max=3.):
        """The constructor for the ImageDisp class.

        Inputs:

        imgdata_list
            A list of 2-D (height x width) numpy arrays, each containing a
            single image's data. May also be a single array instead of a list,
            in which case there is only one image. If there are multiple images,
            they are arranged horizontally with a single set of scrollbars, and
            are all controlled by the same image controls (except see
            separate_contrast). All images must have the same dimensions.

        overlay_list
            An optional list of 2-D (height x width) or 3-D
            (height x width x {3,4}) color overlays, one per image. A list entry
            may be None indicating no overlay for that image, or the entire
            argument may be None indicating no overlays at all.

            The height and width may be an integer multiple of the image
            dimensions. In this case, the image is scaled-up (with each pixel
            replicated in an NxM grid) so that the overlay corresponds with a
            sub-pixel area.

            If the array values are integers, they are assumed to range from
            0-255. Otherwise, the array values should be floating point from
            0.0 - 1.0.

            If the overlay is 2-D, the overlay is red. Otherwise, the third
            dimension is either (R,G,B) or (R,G,B,A). If present, the alpha
            value (A) represents the transparency of the overlay, with 0.0 being
            invisible and 1.0 being opaque. This alpha level is combined with
            the overlay transparency slider to produce the final transparency
            value. If no alpha channel is provided, the overlay is opaque except
            where it is (0,0,0), in which case it is transparent.

        color_column_list
            An optional list of 2-D (width x 3) colors. Array values should be
            floating point. The third dimension is (R,G,B). Each vertical slice
            of an image is multiplied by the corresponding RGB values in the
            column. This is useful for tinting each column as a single color
            without having to specify a full overlay. Note, however, that the
            overlay transparency slider has no effect and there is no alpha
            channel.

        parent
            The parent Frame, if any. None means to use the Toplevel widget.
            If parent is provided, the title is not set and the window destroy
            action is not registered.

        title
            The title of the window.

        canvas_size
            The size (width, height) of the Canvas widget to display the image.
            The Canvas will always be this size, regardless of the size of the
            image. If the image at the current zoom level is larger than the
            canvas, scrollbars will appear. Defaults to the full size of the
            image.

        flip_y
            By default (0,0) is in the top left corner of the display. If flip_y
            is True then the image is flipped across a horizontal line and (0,0)
            is at the bottom left corner of the display.

        origin
            An optional tuple (x,y) giving the pixel location of the origin
            (0,0). This is used to display the mouse pointer location and to
            adjust the pixel location passed to callbacks.

        enlarge_limit
            The maximum magnification permitted beyond unit zoom. This can be a
            tuple (max_x, max_y).

        shrink_limit
            The maximum shrink permitted beyond unit zoom. This can be a
            tuple (max_x, max_y).

        one_zoom
            True to provide only one zoom slider that affects both x and y
            equally. False to provide two sliders to allow zoom for x and y
            separately.

        auto_update
            The initial setting of the Auto Update checkbox. True to refresh
            the image instantly whenever a slider is moved.

        overlay_transparency
            The initial value for the overlay transparency slider.

        blackpoint
            The initial blackpoint setting. None to use the minimum value from
            all images (or each image).

        blackpoint_min
            The minimum value for the blackpoint slider. None to use the minimum
            value from all images.

        blackpoint_max
            The maximum value for the blackpoint slider. None to use the maximum
            value from all images.

        whitepoint
            The initial whitepoint setting. None to use the maximum value from
            all images as adjusted by whitepoint_ignore_frac.

        whitepoint_min
            The minimum value for the whitepoint slider. None to use the minimum
            value from all images.

        whitepoint_max
            The maximum value for the whitepoint slider. None to use the maximum
            value from all images.

        whitepoint_ignore_frac
            The percentile to use to determine the maximum whitepoint. 1.0 means
            to use the brightest datum in any image.

        gamma
            The initial gamma setting.

        gamma_min
            The minimum value for the gamma slider.

        gamma_max
            The maximum value for the gamma slider.

        separate_contrast
            True to allow separate blackpoint/whitepoint/gamma for each image.
            In this case, each of the above parameters is actually a list of
            values, one per image.
        """

        if parent:
            self._toplevel = parent
        else:
            self._toplevel = tk.Toplevel()
            if title is not None:
                self._toplevel.title(title)

        ### Init the Tk Frame
        tk.Frame.__init__(self, self._toplevel)
        if not parent:
            self._toplevel.protocol('WM_DELETE_WINDOW', self._command_wm_delete)

        if not isinstance(imgdata_list, Sequence):
            imgdata_list = [imgdata_list]

        if canvas_size is None:
            imheight, imwidth = imgdata_list[0].shape
        else:
            imwidth, imheight = canvas_size

        self._canvas_size_x = imwidth
        self._canvas_size_y = imheight
        self._flip_y = flip_y
        self._origin = origin

        self._separate_contrast = separate_contrast
        if not isinstance(blackpoint, Sequence):
            blackpoint = [blackpoint] * len(imgdata_list)
        if not isinstance(whitepoint, Sequence):
            whitepoint = [whitepoint] * len(imgdata_list)
        if not isinstance(whitepoint_ignore_frac, Sequence):
            whitepoint_ignore_frac = ([whitepoint_ignore_frac] *
                                      len(imgdata_list))
        if not isinstance(gamma, Sequence):
            gamma = [gamma] * len(imgdata_list)
        if not isinstance(gamma_min, Sequence):
            gamma_min = [gamma_min] * len(imgdata_list)
        if not isinstance(gamma_max, Sequence):
            gamma_max = [gamma_max] * len(imgdata_list)
        self._override_blackpoint = blackpoint
        self._override_whitepoint = whitepoint
        self._whitepoint_ignore_frac = whitepoint_ignore_frac
        self._gamma = gamma
        self._gamma_min = gamma_min
        self._gamma_max = gamma_max

        ### Construct the canvases

        self._canvas_list = []

        # Frame for all of the canvases and control areas arranged horizontally
        section_frame = tk.Frame(self)
        # section_frame row 0 => canvases and scrollbars
        for i in range(len(imgdata_list)):
            canvas_frame = tk.Frame(section_frame)
            canvas_frame.grid(row=0, column=i)
            canvas = tk.Canvas(canvas_frame, width=imwidth, height=imheight,
                               bg='yellow', cursor='crosshair')
            canvas.grid(row=0, column=0, sticky='nw')
            self._canvas_list.append(canvas)

            if i == 0:
                # Create scrollbars for the first image only
                self._vert_sbar = tk.Scrollbar(canvas_frame,
                                               orient=tk.VERTICAL,
                                               command=self._command_scroll_y)
                canvas.config(yscrollcommand=self._vert_sbar.set)
                self._vert_sbar.grid(row=0, column=1, sticky='ns')
                self._horiz_sbar = tk.Scrollbar(canvas_frame,
                                                orient=tk.HORIZONTAL,
                                                command=self._command_scroll_x)
                canvas.config(xscrollcommand=self._horiz_sbar.set)
                self._horiz_sbar.grid(row=1, column=0, sticky='ew')

            # Register the mouse motion callback
            canvas.bind("<Motion>",
                        lambda event, img_num=i:
                        self._mousemove_callback_handler(event,
                                                         img_num, None))

        # canvases_frame.pack(side=tk.TOP)

        ### Construct the control sliders and buttons

        self._var_blackpoint = []
        self._var_whitepoint = []
        self._var_gamma = []
        self._last_blackpoint = []
        self._last_whitepoint = []
        self._last_gamma = []
        self._scale_blackpoint = []
        self._scale_whitepoint = []
        self._scale_gamma = []

        for i in range(len(imgdata_list)):
            controls_parent_frame = tk.Frame(section_frame)
            # section_frame row 1 => controls
            controls_parent_frame.grid(row=1, column=i, sticky='nw')
            control_frame = tk.Frame(controls_parent_frame)

            if i == 0:
                self._var_auto_update = tk.IntVar()
                self._var_display_img_overlay = tk.IntVar()
                self._var_overlay_transparency = tk.DoubleVar()
                self._var_overlay_transparency.set(overlay_transparency)

                self._last_display_img_overlay = None
                self._last_overlay_transparency = None
                self._one_zoom = one_zoom

            gridrow = 0

            if i == 0:
                cbutton = tk.Checkbutton(control_frame, text='Auto Update',
                                         variable=self._var_auto_update,
                                         command=self._command_auto_update_checkbox)
                self._var_auto_update.set(auto_update)
                cbutton.grid(row=gridrow, column=0, sticky='w')
                self._button_update = tk.Button(control_frame, text='Update Now',
                                                command=self._command_update_now)
                self._button_update.grid(row=gridrow, column=1)
            gridrow += 1

            if i == 0:
                cbutton = tk.Checkbutton(control_frame, text='Overlay',
                                         variable=self._var_display_img_overlay,
                                         command=self._command_refresh_overlay_checkbox)
                self._var_display_img_overlay.set(1)
                cbutton.grid(row=gridrow, column=0, sticky='w')
                self._scale_overlay = tk.Scale(control_frame,
                                               orient=tk.HORIZONTAL,
                                               resolution=0.05,
                                               from_=0., to_=1.,
                                               variable=self._var_overlay_transparency,
                                               command=
                                                   self._command_refresh_transparency)
                self._scale_overlay.grid(row=gridrow, column=1)
            gridrow += 1

            self._last_blackpoint.append(None)
            if i == 0 or self._separate_contrast:
                var = tk.DoubleVar()
                scale = tk.Scale(control_frame,
                                 orient=tk.HORIZONTAL,
                                 resolution=0.001,
                                 variable=var,
                                 command=self._command_refresh_image_scales)
                label = tk.Label(control_frame, text='Blackpoint')
                label.grid(row=gridrow, column=i*2, sticky='w')
                scale.grid(row=gridrow, column=i*2+1)
            else:
                var = self._var_blackpoint[0]
                scale = self._scale_blackpoint[0]
            self._var_blackpoint.append(var)
            self._scale_blackpoint.append(scale)
            gridrow += 1

            self._last_whitepoint.append(None)
            if i == 0 or self._separate_contrast:
                var = tk.DoubleVar()
                scale = tk.Scale(control_frame,
                                 orient=tk.HORIZONTAL,
                                 resolution=0.001,
                                 variable=var,
                                 command=self._command_refresh_image_scales)
                label = tk.Label(control_frame, text='Whitepoint')
                label.grid(row=gridrow, column=i*2, sticky='w')
                scale.grid(row=gridrow, column=i*2+1)
            else:
                var = self._var_whitepoint[0]
                scale = self._scale_whitepoint[0]
            self._var_whitepoint.append(var)
            self._scale_whitepoint.append(scale)
            gridrow += 1

            self._last_gamma.append(None)
            if i == 0 or self._separate_contrast:
                var = tk.DoubleVar()
                scale = tk.Scale(control_frame,
                                 orient=tk.HORIZONTAL,
                                 resolution=0.01,
                                 from_=self._gamma_min[i],
                                 to=self._gamma_max[i],
                                 variable=var,
                                 command=self._command_refresh_image_scales)
                label = tk.Label(control_frame, text='Gamma')
                label.grid(row=gridrow, column=i*2, sticky='w')
                scale.grid(row=gridrow, column=i*2+1)
            else:
                var = self._var_gamma[0]
                scale = self._scale_gamma[0]
            self._var_gamma.append(var)
            self._scale_gamma.append(scale)
            gridrow += 1

            if i == 0:
                xmin = ymin = None
                self._enlarge_limit_x = None
                self._enlarge_limit_y = None
                if enlarge_limit is not None:
                    if isinstance(enlarge_limit, (tuple, list)):
                        xmin, ymin = enlarge_limit[0], enlarge_limit[1]
                        if xmin is not None:
                            xmin = -xmin
                        if ymin is not None:
                            ymin = -ymin
                    else:
                        xmin = -enlarge_limit
                        ymin = -enlarge_limit
                    self._enlarge_limit_x = xmin
                    self._enlarge_limit_y = ymin
                if xmin is None:
                    xmin = 0
                if ymin is None:
                    ymin = 0

                xmax = ymax = None
                self._shrink_limit_x = None
                self._shrink_limit_y = None
                img_xmax = int(max(np.ceil(
                    float(imgdata_list[0].shape[1])/imwidth), 1)-1)
                img_ymax = int(max(np.ceil(
                    float(imgdata_list[0].shape[0])/imheight), 1)-1)
                if shrink_limit is not None:
                    if isinstance(shrink_limit, (tuple, list)):
                        xmax, ymax = shrink_limit[0], shrink_limit[1]
                    else:
                        xmax = shrink_limit
                        ymax = shrink_limit
                    self._shrink_limit_x = xmax
                    self._shrink_limit_y = ymax
                    if xmax is not None and img_xmax > xmax:
                        xmax = img_xmax
                    if ymax is not None and img_ymax > ymax:
                        ymax = img_ymax
                if xmax is None:
                    xmax = img_xmax
                if ymax is None:
                    ymax = img_ymax

                if one_zoom:
                    self._var_zoom = tk.IntVar()
                    self._last_zoom = None

                    label = tk.Label(control_frame, text='Zoom')
                    label.grid(row=gridrow, column=0, sticky='w')
                    xymax = max(xmax, ymax)
                    xymin = min(xmin, ymin)
                    self._scale_zoom = tk.Scale(control_frame,
                                            orient=tk.HORIZONTAL,
                                            from_=xymin, to=xymax,
                                            variable=self._var_zoom,
                                            command=self._command_refresh_zoom)
                    self._var_zoom.set(xymax)
                    self._scale_zoom.grid(row=gridrow, column=1)
                    gridrow += 1
                else:
                    self._var_xzoom = tk.IntVar()
                    self._var_yzoom = tk.IntVar()
                    self._last_xzoom = None
                    self._last_yzoom = None

                    label = tk.Label(control_frame, text='X Zoom')
                    label.grid(row=gridrow, column=0, sticky='w')
                    self._scale_xzoom = tk.Scale(control_frame,
                                                 orient=tk.HORIZONTAL,
                                                 from_=xmin, to=xmax,
                                                 variable=self._var_xzoom,
                                                 command=self._command_refresh_zoom)
                    self._var_xzoom.set(img_xmax)
                    self._scale_xzoom.grid(row=gridrow, column=1)
                    gridrow += 1

                    label = tk.Label(control_frame, text='Y Zoom')
                    label.grid(row=gridrow, column=0, sticky='w')
                    self._scale_yzoom = tk.Scale(control_frame,
                                                 orient=tk.HORIZONTAL,
                                                 from_=ymin, to=ymax,
                                                 variable=self._var_yzoom,
                                                 command=self._command_refresh_zoom)
                    self._var_yzoom.set(img_ymax)
                    self._scale_yzoom.grid(row=gridrow, column=1)
                    gridrow += 1
                self._label_xy = tk.Label(control_frame, text='Mouse coord:')
                self._label_xy.grid(row=gridrow, column=0, columnspan=2, sticky='w')
                gridrow += 1

                self._label_val = tk.Label(control_frame, text='Mouse val:')
                self._label_val.grid(row=gridrow, column=0, columnspan=2, sticky='w')

            if i == 0:
                ### Make another frame for other programs to add their own controls

                self.addon_control_frame = tk.Frame(controls_parent_frame)
                self.addon_control_frame.pack(side=tk.RIGHT, anchor='ne')

            control_frame.pack(side=tk.LEFT, anchor='nw', fill='x', expand=True)
            # controls_parent_frame.pack(anchor='w', fill='x', expand=True)

        section_frame.pack()
        self.pack()
        self.update_image_data(imgdata_list, overlay_list, color_column_list)


    #==========================================================================
    # PUBLIC METHODS
    #==========================================================================

    def update_image_data(self, imgdata_list, overlay_list=None,
                          color_column_list=None, recompute_scales=True):
        """Update the image data with new arrays.

        Inputs:

        imagedata_list     See the ImageDisp constructor.
        overlay_list       See the ImageDisp constructor.
        color_column_list  See the ImageDisp constructor.
        recompute_scales   True to recompute blackpoint, whitepoint, and gamma
                           limits and current slider values. False to leave the
                           current settings alone.
        """

        if not isinstance(imgdata_list, Sequence):
            imgdata_list = [imgdata_list]

        assert len(imgdata_list) == len(self._canvas_list)

        # Replace the overlay list
        if overlay_list is None:
            self._overlay_list = [None] * len(imgdata_list)
        else:
            if (type(overlay_list) != type([]) and
                type(overlay_list) != type(())):
                overlay_list = [overlay_list]
            new_overlay_list = []
            for overlay in overlay_list:
                if overlay is None:
                    new_overlay_list.append(overlay)
                    continue
                if issubclass(overlay.dtype.type, np.integer):
                    overlay = overlay.astype('float') / 255.
                if len(overlay.shape) == 2:
                    new_overlay = np.zeros(overlay.shape + (3,))
                    new_overlay[:,:,0] = overlay
                    new_overlay_list.append(new_overlay)
                else:
                    new_overlay_list.append(overlay)
            self._overlay_list = new_overlay_list

        assert len(imgdata_list) == len(self._overlay_list)

        self._update_overlay_alpha()

        # Scale each image to the size of the corresponding overlay
        new_imgdata_list = []
        overlay_scale_x_list = []
        overlay_scale_y_list = []

        for i in range(len(imgdata_list)):
            if (self._overlay_list[i] is None or
                self._overlay_list[i].shape[:2] == imgdata_list[i].shape):
                new_imgdata_list.append(imgdata_list[i])
                overlay_scale_x_list.append(1)
                overlay_scale_y_list.append(1)
                continue
            scale_x = self._overlay_list[i].shape[1] / imgdata_list[i].shape[1]
            scale_y = self._overlay_list[i].shape[0] / imgdata_list[i].shape[0]
            zoomed_img = ndinterp.zoom(imgdata_list[i], (scale_y, scale_x),
                                       order=0)
            new_imgdata_list.append(zoomed_img)
            overlay_scale_x_list.append(scale_x)
            overlay_scale_y_list.append(scale_y)

        imgdata_list = new_imgdata_list

        self._imgdata_list = imgdata_list
        self._overlay_scale_x_list = overlay_scale_x_list
        self._overlay_scale_y_list = overlay_scale_y_list

        self._update_scaled_images()

        # Replace the color column list
        if color_column_list is None:
            self._color_column_list = [None] * len(imgdata_list)
        else:
            if (type(color_column_list) != type([]) and
                type(color_column_list) != type(())):
                color_column_list = [color_column_list]
            self._color_column_list = color_column_list

        assert len(imgdata_list) == len(self._color_column_list)

        # Recompute the whitepoint/blackpoint/gamma
        if recompute_scales:
            if self._separate_contrast:
                for i, img in enumerate(imgdata_list):
                    if self._whitepoint_ignore_frac == 1.0:
                        themax = img.max()
                    else:
                        img_sorted = np.sort(img, axis=None)
                        themax = img_sorted[
                            np.clip(int(len(img_sorted)*
                                        self._whitepoint_ignore_frac[i]),
                                    0, len(img_sorted)-1)]
                    themin = np.min(img.min())

                    self._scale_blackpoint[i].config(from_=themin,
                                                     to=themax,
                                                     resolution=(themax-themin)/
                                                                1000.)
                    self._scale_whitepoint[i].config(from_=themin,
                                                     to=themax,
                                                     resolution=(themax-themin)/
                                                                1000.)
                    blackpoint = themin
                    if self._override_blackpoint[i] is not None:
                        blackpoint = self._override_blackpoint[i]
                    whitepoint = themax
                    if self._override_whitepoint[i] is not None:
                        whitepoint = self._override_whitepoint[i]
                    self._var_blackpoint[i].set(blackpoint)
                    self._var_whitepoint[i].set(whitepoint)
                    self._var_gamma[i].set(self._gamma[i])
            else:
                imgdata_min_list = []
                imgdata_max_list = []
                for i, img in enumerate(imgdata_list):
                    imgdata_min_list.append(img.min())
                    if self._whitepoint_ignore_frac == 1.0:
                        imgdata_max_list.append(img.max())
                    else:
                        img_sorted = np.sort(img, axis=None)
                        perc_wp = img_sorted[
                            np.clip(int(len(img_sorted)*
                                        self._whitepoint_ignore_frac[i]),
                                    0, len(img_sorted)-1)]
                        imgdata_max_list.append(perc_wp)
                themin = np.min(imgdata_min_list)
                themax = np.max(imgdata_max_list)
                for i in range(len(imgdata_list)):
                    self._scale_blackpoint[i].config(from_=themin,
                                                     to=themax,
                                                     resolution=(themax-themin)/
                                                                1000.)
                    self._scale_whitepoint[i].config(from_=themin,
                                                     to=themax,
                                                     resolution=(themax-themin)/
                                                                1000.)
                blackpoint = themin
                if self._override_blackpoint[0] is not None:
                    blackpoint = self._override_blackpoint[0]
                whitepoint = themax
                if self._override_whitepoint[0] is not None:
                    whitepoint = self._override_whitepoint[0]
                self._var_blackpoint[0].set(blackpoint) # All the same
                self._var_whitepoint[0].set(whitepoint)
                self._var_gamma[0].set(self._gamma[0])

        # Update zoom limits
        xmax = int(max(np.ceil(
            float(imgdata_list[0].shape[1])/self._canvas_size_x), 1)-1)
        ymax = int(max(np.ceil(
            float(imgdata_list[0].shape[0])/self._canvas_size_y), 1)-1)
        if self._shrink_limit_x is not None:
            xmax = max(xmax, self._shrink_limit_x)
        if self._shrink_limit_y is not None:
            ymax = max(ymax, self._shrink_limit_y)
        if self._one_zoom:
            xymax = max(xmax, ymax)
            self._scale_zoom.config(to=xymax)
        else:
            self._scale_xzoom.config(to=xmax)
            self._scale_yzoom.config(to=ymax)

        # Update the display
        self._update_internal_data()
        self._update_pil_images()

    def set_overlay(self, num, overlay):
        """Replace one overlay and redraw the image.

        Inputs:

        num                The image number to replace the overlay of.
        overlay            The new overlay data. See the ImageDisp constructor
                           for full details.
        """
        self._overlay_list[num] = overlay
        self.update_image_data(self._imgdata_list, self._overlay_list,
                               self._color_column_list, recompute_scales=False)

    def set_color_column(self, num, color_column):
        """Replace one color column and redraw the image.

        Inputs:

        num                The image number to replace the overlay of.
        color_column       The new color column data. See the ImageDisp
                           constructor for full details.
        """
        self._color_column_list[num] = color_column
        self.update_image_data(self._imgdata_list, self._overlay_list,
                               self._color_column_list, recompute_scales=False)

    def set_image_params(self, blackpoint, whitepoint, gamma):
        """Force setting of blackpoint, whitepoint, and gamma.

        Inputs:

        blackpoint         The new value for the blackpoint slider.
        whitepoint         The new value for the whitepoint slider.
        gamma              The new value for the gamma slider.

        Note that the minimum and maximum values for each slider remains
        unchanged.
        """
        self._var_blackpoint.set(blackpoint)
        self._var_whitepoint.set(whitepoint)
        self._var_gamma.set(gamma)

        if (self._var_blackpoint.get() == self._last_blackpoint and
            self._var_whitepoint.get() == self._last_whitepoint and
            self._var_gamma.get() == self._last_gamma):
            return

        self._last_blackpoint = self._var_blackpoint.get()
        self._last_whitepoint = self._var_whitepoint.get()
        self._last_gamma = self._var_gamma.get()
        self._update_scaled_images()
        self._update_pil_images()

    @staticmethod
    def scale_image(img, blackpoint, whitepoint, gamma):
        """Scale a 2-D image based on blackpoint, whitepoint, and gamma.

        This is a generic routine that can be used outside of the scope of
        this class.

        Inputs:

        img                The 2-D image.
        blackpoint         Any element below the blackpoint will be black.
        whitepoint         Any element above the whitepoint will be white.
        gamma              Non-linear stretch (1.0 = linear stretch).
        """

        if whitepoint < blackpoint:
            whitepoint = blackpoint

        if whitepoint == blackpoint:
            whitepoint += 0.00001

        greyscale_img = np.floor((np.maximum(img-blackpoint, 0)/
                                  (whitepoint-blackpoint))**gamma*256)
        greyscale_img = np.clip(greyscale_img, 0, 255) # Clip black and white
        return greyscale_img

    def bind_mousemove(self, img_num, callback_func):
        """Bind a mouse move callback for a single image.

        Inputs:

        img_num            The number of the image
        callback_func      The function to be called on mouse move; it is
                           called with params (x, y) in image (not screen
                           pixel) coordinates.
        """
        canvas = self._canvas_list[img_num]
        canvas.bind("<Motion>",
                    lambda event, callback_func=callback_func, img_num=img_num:
                    self._mousemove_callback_handler(event, img_num,
                                                     callback_func))

    def bind_b1press(self, img_num, callback_func):
        """Bind a button-one callback for a single image.

        Inputs:

        img_num            The number of the image
        callback_func      The function to be called on mouse move; it is
                           called with params (x, y) in image (not screen
                           pixel) coordinates.
        """
        canvas = self._canvas_list[img_num]
        canvas.bind("<Button-1>",
                    lambda event, callback_func=callback_func, img_num=img_num:
                    self._b1press_callback_handler(event, img_num,
                                                   callback_func))

    def bind_ctrl_b1press(self, img_num, callback_func):
        """Bind a Control+button-one callback for a single image.

        Inputs:

        img_num            The number of the image
        callback_func      The function to be called on mouse move; it is
                           called with params (x, y) in image (not screen
                           pixel) coordinates.
        """

        canvas = self._canvas_list[img_num]
        canvas.bind("<Control-Button-1>",
                    lambda event, callback_func=callback_func, img_num=img_num:
                    self._b1press_ctrl_callback_handler(event, img_num,
                                                        callback_func))

    def bind_shift_b1press(self, img_num, callback_func):
        """Bind a Shift+button-one callback for a single image.

        Inputs:

        img_num            The number of the image
        callback_func      The function to be called on mouse move; it is
                           called with params (x, y) in image (not screen
                           pixel) coordinates.
        """

        canvas = self._canvas_list[img_num]
        canvas.bind("<Shift-Button-1>",
                    lambda event, callback_func=callback_func, img_num=img_num:
                    self._b1press_shift_callback_handler(event, img_num,
                                                         callback_func))


    #==========================================================================
    # INTERNAL METHODS
    #==========================================================================

    def _get_zoom_factors(self):
        """Internal - Get the X and Y zoom factors accounting for single or
        double scale bars.
        """
        if self._one_zoom:
            zoom = float(self._var_zoom.get()+1)
            if zoom <= 0:
                zoom = 1./(abs(zoom)+2)
            return zoom, zoom

        xzoom = float(self._var_xzoom.get()+1)
        if xzoom <= 0:
            xzoom = 1./(abs(xzoom)+2)
        yzoom = self._var_yzoom.get()+1
        if yzoom <= 0:
            yzoom = 1./(abs(yzoom)+2)

        return xzoom, yzoom

    def _update_overlay_alpha(self):
        """Internal - Update the overlay alpha channel.

        We only compute this as necessary to increase the performance of
        _update_pil_images.
        """
        self._one_minus_alpha3d_list = []
        self._scaled_alpha3d_list = []

        for overlay in self._overlay_list:
            if overlay is None:
                self._one_minus_alpha3d_list.append(None)
                self._scaled_alpha3d_list.append(None)
                continue
            if overlay.shape[2] == 4:
                # User-specified alpha channel
                alpha = overlay[:,:,3]
            else:
                # Alpha channel for any pixel where the overlay isn't zero
                overlay_max = np.max(overlay, axis=2)
                alpha = overlay_max != 0
            alpha3d = np.empty(alpha.shape+(3,))
            alpha3d[:,:,:] = alpha[:,:,np.newaxis]
            alpha3d *= self._var_overlay_transparency.get()
            self._one_minus_alpha3d_list.append(1-alpha3d)
            self._scaled_alpha3d_list.append(255*alpha3d)

    def _update_scaled_images(self):
        """Internal - Update the scaled images.

        We only compute this as necessary to increase the performance of
        _update_pil_images.
        """
        self._greyscale_list = []
        for i, img in enumerate(self._imgdata_list):
            greyscale_img = ImageDisp.scale_image(img,
                                                  self._var_blackpoint[i].get(),
                                                  self._var_whitepoint[i].get(),
                                                  self._var_gamma[i].get())
            self._greyscale_list.append(greyscale_img)

    def _update_internal_data(self):
        """Update all precomputed data.
        """
        self._last_overlay_transparency = self._var_overlay_transparency.get()
        self._update_overlay_alpha()

        for i in range(len(self._imgdata_list)):
            self._last_blackpoint[i] = self._var_blackpoint[i].get()
            self._last_whitepoint[i] = self._var_whitepoint[i].get()
            self._last_gamma[i] = self._var_gamma[i].get()
        self._update_scaled_images()

        if self._one_zoom:
            self._last_zoom = self._var_zoom.get()
        else:
            self._last_xzoom = self._var_xzoom.get()
            self._last_yzoom = self._var_yzoom.get()

    def _update_pil_images(self):
        """Internal - update the underlying PIL images with the current images,
        overlays, and image settings.
        """
        scroll_x = self._horiz_sbar.get()
        scroll_x_ctr = (scroll_x[0] + scroll_x[1]) / 2
        scroll_y = self._vert_sbar.get()
        scroll_y_ctr = (scroll_y[0] + scroll_y[1]) / 2

        # Convert the FP image data to integers 0-255
        self._displayed_img_list = []
        self._pim_list = []
        first_im = None
        for i, img in enumerate(self._imgdata_list):
            greyscale_img = self._greyscale_list[i]
            cur_overlay = self._overlay_list[i]
            if self._var_display_img_overlay.get() == 0:
                cur_overlay = None
            if (self._color_column_list[i] is not None or
                cur_overlay is not None):
                mode = 'RGB'
                combined_data = np.zeros(greyscale_img.shape + (3,))
                combined_data[:,:,:] = greyscale_img[:,:,np.newaxis]

                color_column = self._color_column_list[i]
                if color_column is not None:
                    alpha = self._var_overlay_transparency.get()
                    combined_data *= (
                        (1-alpha) + color_column[np.newaxis,:,:] * alpha)

                if cur_overlay is not None:
                    combined_data = (combined_data*
                                     self._one_minus_alpha3d_list[i] +
                                     cur_overlay[:,:,:3]*
                                     self._scaled_alpha3d_list[i])

                combined_data = np.cast['uint8'](combined_data)

                if self._flip_y:
                    combined_data = combined_data[::-1,:,:]+0
                    # +0 forces a copy, necessary for PIL
            else:
                mode = 'L'
                # This copying also makes it contiguous, which is necessary for
                # PIL
                combined_data = np.zeros(greyscale_img.shape, dtype=np.uint8)
                combined_data[:,:] = greyscale_img
                if self._flip_y:
                    combined_data = combined_data[::-1,:]+0
                    # +0 forces a copy, necessary for PIL
            # Necessary to prevent garbage collection
            self._displayed_img_list.append(combined_data)
            imheight, imwidth = combined_data.shape[:2]
            im = Image.frombuffer(mode, (imwidth, imheight), combined_data,
                                  'raw', mode, 0, 1)
            xzoom, yzoom = self._get_zoom_factors()
            if xzoom != 1 or yzoom != 1:
                im = im.resize((int(imwidth/xzoom), int(imheight/yzoom)),
                               Image.Resampling.NEAREST)
            if i == 0:
                first_im = im
            pim = ImageTk.PhotoImage(im)
            # Necessary to prevent garbage collection
            self._pim_list.append(pim)
            if len(self._canvas_list[i].find_withtag('img')) == 0:
                self._canvas_list[i].create_image(0, 0, image=pim, anchor='nw',
                                                 tags='img')
                self._canvas_list[i].config(scrollregion=(0, 0,
                                                         im.size[0],
                                                         im.size[1]))
            else:
                self._canvas_list[i].itemconfig('img', image=pim)
                self._canvas_list[i].config(scrollregion=(0, 0,
                                                         im.size[0],
                                                         im.size[1]))

        scroll_x_width = float(self._canvas_size_x) / first_im.size[0]
        scroll_x_min = scroll_x_ctr - scroll_x_width / 2
        scroll_x_max = scroll_x_ctr + scroll_x_width / 2
        if scroll_x_min < 0:
            scroll_x_max += -scroll_x_min
            scroll_x_min = 0
        if scroll_x_max > 1:
            scroll_x_min -= (scroll_x_max-1)
            scroll_x_max = 1

        scroll_y_width = float(self._canvas_size_y) / first_im.size[1]
        scroll_y_min = scroll_y_ctr - scroll_y_width / 2
        scroll_y_max = scroll_y_ctr + scroll_y_width / 2
        if scroll_y_min < 0:
            scroll_y_max += -scroll_y_min
            scroll_y_min = 0
        if scroll_y_max > 1:
            scroll_y_min -= (scroll_y_max-1)
            scroll_y_max = 1

        for canvas in self._canvas_list:
            canvas.xview_moveto(scroll_x_min)
            canvas.yview_moveto(scroll_y_min)

    def _command_wm_delete(self):
        """Internal - callback for window manager closing window."""
        self._toplevel.destroy()
        self._toplevel.quit()

    def _command_update_now(self):
        """Internal - callback for update now button."""
        update1 = self._update_transparency()
        update2 = self._update_image_scales()
        update3 = self._update_zoom()
        if update1 or update2 or update3:
            self._update_pil_images()

    def _command_auto_update_checkbox(self):
        """Internal - callback for auto update checkbox."""
        if not self._var_auto_update.get():
            return

        update1 = self._update_transparency()
        update2 = self._update_image_scales()
        update3 = self._update_zoom()
        if update1 or update2 or update3:
            self._update_pil_images()

    def _update_transparency(self):
        """Internal - update for change in transparency."""
        if (self._var_display_img_overlay.get() ==
            self._last_display_img_overlay and
            self._var_overlay_transparency.get() ==
            self._last_overlay_transparency):
            return False

        self._last_display_img_overlay = self._var_display_img_overlay.get()
        self._last_overlay_transparency = self._var_overlay_transparency.get()
        self._update_overlay_alpha()
        return True

    def _command_refresh_overlay_checkbox(self):
        """Internal - callback for overlay checkbox."""
        if not self._var_auto_update.get():
            return
        if self._update_transparency():
            self._update_pil_images()

    def _command_refresh_transparency(self, val):
        """Internal - callback for transparency slider motion."""
        if not self._var_auto_update.get():
            return
        if not self._var_display_img_overlay.get():
            return
        if self._update_transparency():
            self._update_pil_images()

    def _update_image_scales(self):
        """Internal - update for change in image scales."""
        did_something = False
        for i in range(len(self._imgdata_list)):
            if (self._var_blackpoint[i].get() == self._last_blackpoint[i] and
                self._var_whitepoint[i].get() == self._last_whitepoint[i] and
                self._var_gamma[i].get() == self._last_gamma[i]):
                continue

            self._last_blackpoint[i] = self._var_blackpoint[i].get()
            self._last_whitepoint[i] = self._var_whitepoint[i].get()
            self._last_gamma[i] = self._var_gamma[i].get()
            self._update_scaled_images()
            did_something = True

        return did_something

    def _command_refresh_image_scales(self, val):
        """Internal - callback for image scale slider motion."""
        if not self._var_auto_update.get():
            return
        if self._update_image_scales():
            self._update_pil_images()

    def _update_zoom(self):
        """Internal - update for change in zoom."""
        if ((self._one_zoom and
             self._var_zoom.get() == self._last_zoom) or
            (not self._one_zoom and
             (self._var_xzoom.get() == self._last_xzoom and
              self._var_yzoom.get() == self._last_yzoom))):
            return False

        xzoom, yzoom = self._get_zoom_factors()

        for i, img in enumerate(self._imgdata_list):
            greyscale_img = self._greyscale_list[i]
            num_pix = (greyscale_img.shape[0]*greyscale_img.shape[1]/
                       xzoom/yzoom)
#             if num_pix > 5760000:
#                 return False

        if self._one_zoom:
            self._last_zoom = self._var_zoom.get()
        else:
            self._last_xzoom = self._var_xzoom.get()
            self._last_yzoom = self._var_yzoom.get()

        return True

    def _command_refresh_zoom(self, val):
        """Internal - callback for zoom slider motion."""
        if not self._var_auto_update.get():
            return
        if self._update_zoom():
            self._update_pil_images()

    def _command_scroll_x(self, *args):
        """Internal - callback for X scrollbar motion."""
        for canvas in self._canvas_list:
            canvas.xview(*args)

    def _command_scroll_y(self, *args):
        """Internal - callback for Y scrollbar motion."""
        for canvas in self._canvas_list:
            canvas.yview(*args)

    def _mousemove_callback_handler(self, event, img_num, callback_func):
        """Internal - callback for mouse move."""
        if self._imgdata_list[img_num] is None:
            return
        canvas = self._canvas_list[img_num]
        y = canvas.canvasy(event.y)
        xzoom, yzoom = self._get_zoom_factors()
        if self._flip_y:
            y = (self._imgdata_list[img_num].shape[0] /
                 yzoom * self._overlay_scale_y_list[img_num]) - 1 - y
        x = (canvas.canvasx(event.x) * xzoom /
             self._overlay_scale_x_list[img_num])
        y = (y * yzoom /
             self._overlay_scale_y_list[img_num])
        if (x < 0 or y < 0 or
            x >= self._imgdata_list[img_num].shape[1] or
            y >= self._imgdata_list[img_num].shape[0]):
            return
        ndigits_x = len(str(self._imgdata_list[img_num].shape[1]-1))
        ndigits_y = len(str(self._imgdata_list[img_num].shape[0]-1))
        self._label_xy.config(text=('Mouse coord: %'+str(ndigits_x+3)+
                                    '.2f, %'+str(ndigits_y+3)+'.2f') %
                                        (x-self._origin[0], y-self._origin[1]))
        val = self._imgdata_list[img_num][int(y),int(x)]
        if val > 10000:
            self._label_val.config(text='Mouse val: %e' % val)
        else:
            self._label_val.config(text='Mouse val: %12.7f' % val)
        if callback_func is not None:
            callback_func(x-self._origin[0], y-self._origin[1])

    def _b1press_callback_handler(self, event, img_num, callback_func):
        """Internal - callback for button-one press."""
        canvas = self._canvas_list[img_num]
        xzoom, yzoom = self._get_zoom_factors()
        x = (canvas.canvasx(event.x) * xzoom /
             self._overlay_scale_x_list[img_num])
        y = (canvas.canvasy(event.y) * yzoom /
             self._overlay_scale_y_list[img_num])
        if (x < 0 or y < 0 or
            x >= self._imgdata_list[img_num].shape[1] or
            y >= self._imgdata_list[img_num].shape[0]):
            return
        if self._flip_y:
            y = self._imgdata_list[img_num].shape[0] - y -1
        callback_func(x, y)

    def _b1press_ctrl_callback_handler(self, event, img_num, callback_func):
        """Internal - callback for Control+button-one press."""
        canvas = self._canvas_list[img_num]
        xzoom, yzoom = self._get_zoom_factors()
        x = (canvas.canvasx(event.x) * xzoom /
             self._overlay_scale_x_list[img_num])
        y = (canvas.canvasy(event.y) * yzoom /
             self._overlay_scale_y_list[img_num])
        if (x < 0 or y < 0 or
            x >= self._imgdata_list[img_num].shape[1] or
            y >= self._imgdata_list[img_num].shape[0]):
            return
        if self._flip_y:
            y = self._imgdata_list[img_num].shape[0] - y -1
        callback_func(x, y)

    def _b1press_shift_callback_handler(self, event, img_num, callback_func):
        """Internal - callback for Shift+button-one press."""
        canvas = self._canvas_list[img_num]
        xzoom, yzoom = self._get_zoom_factors()
        x = (canvas.canvasx(event.x) * xzoom /
             self._overlay_scale_x_list[img_num])
        y = (canvas.canvasy(event.y) * yzoom /
             self._overlay_scale_y_list[img_num])
        if (x < 0 or y < 0 or
            x >= self._imgdata_list[img_num].shape[1] or
            y >= self._imgdata_list[img_num].shape[0]):
            return
        if self._flip_y:
            y = self._imgdata_list[img_num].shape[0] - y -1
        callback_func(x, y)


#===============================================================================
#
# Validating versions of the Entry widget
#
# From: http://effbot.org/zone/tkinter-entry-validate.htm
#
#===============================================================================

class ValidatingEntry(tk.Entry):
    # base class for validating entry widgets

    def __init__(self, master, value="", **kw):
        super().__init__(master, **kw)
        self.__value = value
        self.__variable = tk.StringVar()
        self.__variable.set(value)
        self.__variable.trace("w", self.__callback)
        self.config(textvariable=self.__variable)

    def __callback(self, *dummy):
        value = self.__variable.get()
        newvalue = self.validate(value)
        if newvalue is None:
            self.__variable.set(self.__value)
        elif newvalue != value:
            self.__value = newvalue
            self.__variable.set(self._newvalue)
        else:
            self.__value = value

    def validate(self, value):
        # override: return value, new value, or None if invalid
        return value

    def value(self):
        return self.__value

class IntegerEntry(ValidatingEntry):
    """A class to implement a Tkinter text-entry widget that only allows integers."""

    def __init__(self, master, value=0, **kw):
        super().__init__(master, str(value), **kw)

    def validate(self, value):
        try:
            if value:
                v = int(value)
            return value
        except ValueError:
            return None

    def value(self):
        return int(super().value())

class FloatEntry(ValidatingEntry):
    """A class to implement a Tkinter text-entry widget that only allows floating point values."""

    def __init__(self, master, value=0., **kw):
        super().__init__(master, str(value), **kw)

    def validate(self, value):
        try:
            if value:
                v = float(value)
            return value
        except ValueError:
            if value == '-' or value == '.' or value == '-.':
                return value
            return None

    def value(self):
        return float(super().value())

#===============================================================================
#
# A scrollable list
#
# From: http://www.nmt.edu/tcc/help/lang/python/examples/scrolledlist
#
#===============================================================================

class ScrolledList(tk.Frame):
    """A compound widget containing a listbox and up to two scrollbars.

      State/invariants:
        .listbox:      [ The Listbox widget ]
        .vScrollbar:
           [ if self._has a vertical scrollbar ->
               that scrollbar
             else -> None ]
        .hScrollbar:
           [ if self._has a vertical scrollbar ->
               that scrollbar
             else -> None ]
        .callback:     [ as passed to constructor ]
        .vscroll:      [ as passed to constructor ]
        .hscroll:      [ as passed to constructor ]
    """

    def __init__(self, master=None, width=40, height=25,
                 vscroll=True, hscroll=False, **kwargs):
        """Constructor for ScrolledList.
        """
        tk.Frame.__init__ (self, master)
        self._width     = width
        self._height    = height
        self._vscroll   = vscroll
        self._hscroll   = hscroll

        if self._vscroll:
            self._vScrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)
            self._vScrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)
        if self._hscroll:
            self._hScrollbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
            self._hScrollbar.grid(row=1, column=0, sticky=tk.E+tk.W)

        self.listbox  =  tk.Listbox(self, relief=tk.SUNKEN,
                                 width=self._width, height=self._height,
                                 borderwidth=2, **kwargs)
        self.listbox.grid(row=0, column=0)
        if self._vscroll:
            self.listbox['yscrollcommand'] = self._vScrollbar.set
            self._vScrollbar['command'] = self.listbox.yview
        if self._hscroll:
            self.listbox['xscrollcommand'] = self._hScrollbar.set
            self._hScrollbar['command'] = self.listbox.xview

    def count(self):
        """Return the number of lines in use in the listbox.
        """
        return self.listbox.size()

    def __getitem__(self, k):
        """Get the (k)th line from the listbox.
        """

        if (0 <= k < self.count()):
            return self.listbox.get(k)
        else:
            raise IndexError("ScrolledList[%d] out of range." % k)

    def append(self, text):
        """Append a line to the listbox.
        """
        self.listbox.insert(tk.END, text)

    def insert(self, linex, text):
        """Insert a line between two existing lines.
        """

        if 0 <= linex < self.count():
            where = linex
        else:
            where = tk.END

        self.listbox.insert(where, text)

    def delete( self, linex ):
        """Delete a line from the listbox.
        """
        if 0 <= linex < self.count():
            self.listbox.delete(linex)

    def clear(self):
        """Remove all lines.
        """
        self.listbox.delete(0, tk.END)

#===============================================================================
#
# Drawing algorithms
#
#===============================================================================

def draw_line(img, x0, y0, x1, y1, color, thickness=1):
    """Draw a line using Bresenham's algorithm with the given thickness.

    The line is drawn by drawing each point as a line perpendicular to
    the main line.

    Input:

        img        The 2-D (or higher) image to draw on.
        x0, y0     The starting point.
        x1, y1     The ending point.
        color      The scalar (or higher) color to draw.
        thickness  The thickness (total width) of the line.
    """
    x0 = int(x0)
    x1 = int(x1)
    y0 = int(y0)
    y1 = int(y1)

    if thickness == 1:
        # Do the simple version
        dx = abs(x1-x0)
        dy = abs(y1-y0)
        if x0 < x1:
            sx = 1
        else:
            sx = -1
        if y0 < y1:
            sy = 1
        else:
            sy = -1
        err = dx-dy

        while True:
            img[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2*err
            if e2 > -dy:
                err = err - dy
                x0 = x0 + sx
            if e2 < dx:
                err = err + dx
                y0 = y0 + sy
        return

    # Find the perpendicular angle
    angle = np.arctan2(y1-y0, x1-x0)
    x_offset = int(np.round(np.cos(angle)))
    y_offset = int(np.round(np.sin(angle)))
    perp_angle = angle + np.pi/2
    perp_x1 = int(np.round(thickness/2.*np.cos(perp_angle)))
    perp_x0 = -perp_x1
    perp_y1 = int(np.round(thickness/2.*np.sin(perp_angle)))
    perp_y0 = -perp_y1
    if perp_x0 == perp_x1 and perp_y0 == perp_y1:
        draw_line(img, color, x0, y0, x1, y1)
        return

    # Compute the perpendicular offsets using one pass of Bresenham's
    perp_offsets_x = []
    perp_offsets_y = []

    dx = abs(perp_x1-perp_x0)
    dy = abs(perp_y1-perp_y0)
    if perp_x0 < perp_x1:
        sx = 1
    else:
        sx = -1
    if perp_y0 < perp_y1:
        sy = 1
    else:
        sy = -1
    err = dx-dy

    while True:
        # There's a better way to do this, but it's patented by IBM!
        # So just do something brute force instead
        perp_offsets_x.append(perp_x0)
        perp_offsets_y.append(perp_y0)
        perp_offsets_x.append(perp_x0+x_offset)
        perp_offsets_y.append(perp_y0)
        perp_offsets_x.append(perp_x0)
        perp_offsets_y.append(perp_y0+y_offset)
        perp_offsets_x.append(perp_x0+x_offset)
        perp_offsets_y.append(perp_y0+y_offset)
        if perp_x0 == perp_x1 and perp_y0 == perp_y1:
            break
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            perp_x0 = perp_x0 + sx
        if e2 < dx:
            err = err + dx
            perp_y0 = perp_y0 + sy

    # Now draw the final line applying the offsets
    dx = abs(x1-x0)
    dy = abs(y1-y0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx-dy

    while True:
        for i in range(len(perp_offsets_x)):
            img[y0+perp_offsets_y[i], x0+perp_offsets_x[i]] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            x0 = x0 + sx
        if e2 < dx:
            err = err + dx
            y0 = y0 + sy

def draw_rect(img, xctr, yctr, xhalfwidth, yhalfwidth, color, thickness=1):
    """Draw a rectangle with the given line thickness.

    Input:

        img        The 2-D (or higher) image to draw on.
        xctr, yctr The center of the rectangle.
        xhalfwidth The width of the rectangle on each side of the center.
        yhalfwidth This is the inner border of the rectangle.
        color      The scalar (or higher) color to draw.
        thickness  The thickness (total width) of the line.
    """
    xctr = int(xctr)
    yctr = int(yctr)
    xhalfwidth = int(xhalfwidth)
    yhalfwidth = int(yhalfwidth)

    # Top
    img[yctr-yhalfwidth-thickness+1:yctr-yhalfwidth+1,
        xctr-xhalfwidth-thickness+1:xctr+xhalfwidth+thickness] = color
    # Bottom
    img[yctr+yhalfwidth:yctr+yhalfwidth+thickness,
        xctr-xhalfwidth-thickness+1:xctr+xhalfwidth+thickness] = color
    # Left
    img[yctr-yhalfwidth-thickness+1:yctr+yhalfwidth+thickness,
        xctr-xhalfwidth-thickness+1:xctr-xhalfwidth+1] = color
    # Right
    img[yctr-yhalfwidth-thickness+1:yctr+yhalfwidth+thickness,
        xctr+xhalfwidth:xctr+xhalfwidth+thickness] = color

def draw_circle(img, x0, y0, r, color, thickness=1):
    """Draw a circle using Bresenham's algorithm with the given thickness.

    Input:

        img        The 2-D (or higher) image to draw on.
        x0, y0     The middle of the circle.
        r          The radius of the circle.
        color      The scalar (or higher) color to draw.
        thickness  The thickness (total width) of the circle.
    """
    def _draw_circle(img, x0, y0, r, color, bigpixel):
        x0 = int(x0)
        y0 = int(y0)
        r = int(r)
        x = -r
        y = 0
        err = 2-2*r
        if bigpixel:
            off_list = [-1, 0, 1]
        else:
            off_list = [0]

        while x < 0:
            for xoff in off_list:
                for yoff in off_list:
                    if (0 <= y0+y+yoff < img.shape[0] and
                        0 <= x0-x+xoff < img.shape[1]):
                        img[y0+y+yoff,x0-x+xoff] = color
                    if (0 <= y0-x+yoff < img.shape[0] and
                        0 <= x0-y+xoff < img.shape[1]):
                        img[y0-x+yoff,x0-y+xoff] = color
                    if (0 <= y0-y+yoff < img.shape[0] and
                        0 <= x0+x+xoff < img.shape[1]):
                        img[y0-y+yoff,x0+x+xoff] = color
                    if (0 <= y0+x+yoff < img.shape[0] and
                        0 <= x0+y+xoff < img.shape[1]):
                        img[y0+x+yoff,x0+y+xoff] = color
            r = err
            if r <= y:
                y = y+1
                err = err+y*2+1
            if r > x or err > y:
                x = x+1
                err = err+x*2+1

    x0 = int(x0)
    y0 = int(y0)
    r = int(r)

    if thickness == 1:
        _draw_circle(img, x0, y0, r, color, False)
        return

    if thickness <= 3:
        _draw_circle(img, x0, y0, r, color, True)
        return

    # This is not perfect, but it's simple
    for r_offset in np.arange(-(thickness-2)/2., (thickness-2)/2.+0.5, 0.5):
        _draw_circle(img, x0, y0, r+r_offset, color, True)

#
#
#

if __name__ == '__main__':
    def callback_move_handler(x, y):
        print('Callback move', x, y)
    def callback_press_handler(x, y):
        print('Callback press', x, y)

    import os
    import sys
    import matplotlib.pyplot as plt

    root = tk.Tk()
    root.withdraw()

    if sys.argv[1] == 'draw':
        # Test drawing algorithms
        print('Drawing tests:')
        print('#1 - Line width 1')
        img = np.zeros((500,500))
        for angle in np.arange(0., 360., 10):
            draw_line(img, 250, 250,
                      250+200*np.cos(angle*np.pi/180),
                      250+200*np.sin(angle*np.pi/180), 1.)
        plt.figure(figsize=(12,12))
        plt.imshow(img, cmap=plt.get_cmap('Greys'))

        print('#2 - Line width 10')
        img = np.zeros((500,500))
        for angle in np.arange(0., 360., 10):
            draw_line(img, 250, 250,
                      250+200*np.cos(angle*np.pi/180),
                      250+200*np.sin(angle*np.pi/180), 1., 10)
        plt.figure(figsize=(12,12))
        plt.imshow(img, cmap=plt.get_cmap('Greys'))

        print('#3 - Circle width 1')
        img = np.zeros((500,500))
        for r in np.arange(3, 200, 30):
            draw_circle(img, 250, 250, r, 1.)
        plt.figure(figsize=(12,12))
        plt.imshow(img, cmap=plt.get_cmap('Greys'))

        print('#4 - Varying thickness 1-10')
        img = np.zeros((750,750))
        n = 0
        thickness = 1
        for r in np.arange(3, 300, 15):
            draw_circle(img, 375, 375, r, 1., thickness)
            thickness += 1
            if thickness > 10:
                thickness = 1

        plt.figure(figsize=(12,12))
        plt.imshow(img, cmap=plt.get_cmap('Greys'))
        plt.show()
        sys.exit(0)

    grad1 = (np.tile(np.linspace(0,1,512),512).reshape((512,512))*
             np.linspace(0,1,512).reshape((512,1)))
    grad2 = (np.tile(np.linspace(0,1,2048),1024).reshape((1024,2048))*
             np.linspace(0,1,1024).reshape((1024,1)))

    if sys.argv[1] == 'img1':
        # Test ImageDisp - simple image
        print('Simple 512x512, two zoom, enlarge (5,2)')
        ImageDisp(grad1, canvas_size=(512,512), flip_y=False)
        ImageDisp(grad1, canvas_size=(512,512), flip_y=True)
        ImageDisp((grad2, grad2), canvas_size=(512,512), one_zoom=False,
                  enlarge_limit=(5,2))
        tk.mainloop()

    if sys.argv[1] == 'img2':
        # Two images with overlay, auto_update, enlarge and separate zooms
        # Flipped y
        print('1 - Overlay left 1, right 0.5')
        overlay1 = np.zeros((grad1.shape[0], grad1.shape[1], 3))
        overlay2 = np.zeros((grad1.shape[0], grad1.shape[1], 3))
        overlay1[100:300,200:400,0] = 1.
        overlay2[200:400,100:300,0] = 0.5
        ImageDisp((grad1, grad1), (overlay1, overlay2),
                  canvas_size=(512,512), auto_update=True,
                  enlarge_limit=(5,5), one_zoom=False)

        print('2 - Overlay transparency Red 1, Green 0.75, Yellow 0.5')
        print('3 - Same, separate contrast')
        # Two images with overlay and transparency, auto_update, enlarge and
        # separate zooms
        overlay1 = np.zeros((grad1.shape[0], grad1.shape[1], 4))
        overlay2 = np.zeros((grad1.shape[0], grad1.shape[1], 4))
        overlay1[100:300,200:400] = (1., 0, 0, 1.)
        overlay2[200:400,100:300] = (0., 0.5, 0., 0.75)

        draw_line(overlay1, 100, 400, 400, 450, (1.,1.,0.5,0.5), 10)
        draw_circle(overlay2, 400, 400, 100, (1.,1.,0.5,0.5), 15)

        imdisp = ImageDisp([grad1, grad1], [overlay1, overlay2],
                           canvas_size=(512,512), auto_update=True,
                           enlarge_limit=(5,5), one_zoom=False)
        imdisp = ImageDisp([grad1, grad1], [overlay1, overlay2],
                           canvas_size=(512,512), auto_update=True,
                           enlarge_limit=(5,5), one_zoom=False,
                           separate_contrast=True)
        tk.mainloop()

    # Two images with overzoomed overlay, auto_update, enlarge and
    # separate zooms
    # overlay1 = np.zeros((iss_nac_obs1.data.shape[0]*3,
    #                      iss_nac_obs1.data.shape[1]*3,
    #                      3))
    # overlay2 = np.zeros((iss_nac_obs1.data.shape[0],
    #                      iss_nac_obs1.data.shape[1],
    #                      3))
    # overlay1[200*3:400*3:4,500*3:700*3:5,1] = 1.

    # overlay2[200:400:4,500:700:5,0] = 1.

    # imdisp = ImageDisp([iss_nac_obs1.data, iss_nac_obs1.data],
    #                    [overlay1, overlay2],
    #                    canvas_size=(512,512), auto_update=True,
    #                    enlarge_limit=(5,5), one_zoom=False)
    # tk.mainloop()


#    overlay = np.zeros((img1.shape[0], img1.shape[1], 3))
#    for i in range(1024):
#        overlay[i,i,0] = 1
#        overlay[i,1023-i,1] = 1
#    imdisp.Bind_mousemove(0, callback_move_handler)
#    imdisp.Bind_b1press(0, callback_press_handler)
#    imdisp.SetOverlay(0, overlay)
#    tk.mainloop()
