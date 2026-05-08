"""TiledImageWidget: efficient tiled rendering of large mosaic images.

Image pixel coordinate convention used throughout:
    pixel_x  0 .. n_long-1   increasing right  = increasing corot longitude
    pixel_y  0 .. n_radii-1  increasing DOWN   = DECREASING radius
                 pixel_y=0           → outer radius (top of display)
                 pixel_y=n_radii-1   → inner radius (bottom of display)

The underlying numpy array has row 0 = INNER radius, so the array is
displayed flipped vertically.  Only the viewport's visible region is ever
rendered (tiled rendering), so arbitrary zoom levels are memory-efficient.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.ma as ma

from PyQt6.QtCore import (
    Qt, QEvent, QPoint, QRect, QSize, pyqtSignal,
)
from PyQt6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QAbstractScrollArea, QRubberBand, QSizePolicy

# Zoom slider maps to log scale: slider 1..1000  →  zoom 0.05x..100x
_ZOOM_LOG_LO = np.log10(0.05)   # ≈ -1.301
_ZOOM_LOG_HI = np.log10(100.0)  # = 2.0


def zoom_to_slider(zoom: float) -> int:
    """Convert zoom value to slider integer 1..1000."""
    log = np.log10(max(zoom, 1e-6))
    pos = (log - _ZOOM_LOG_LO) / (_ZOOM_LOG_HI - _ZOOM_LOG_LO) * 999.0 + 1.0
    return int(round(np.clip(pos, 1, 1000)))


def slider_to_zoom(pos: int) -> float:
    """Convert slider integer 1..1000 to zoom value."""
    log = _ZOOM_LOG_LO + (pos - 1) / 999.0 * (_ZOOM_LOG_HI - _ZOOM_LOG_LO)
    return float(10.0 ** log)


def _nice_tick_values(lo: float, hi: float, max_ticks: int) -> np.ndarray:
    """Return ~max_ticks nice tick values spanning [lo, hi]."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return np.array([])
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    raw = span / max(max_ticks - 1, 1)
    exp = 10.0 ** np.floor(np.log10(max(abs(raw), 1e-30)))
    f = raw / exp
    if f < 1.5:
        step = exp
    elif f < 3.5:
        step = 2.0 * exp
    elif f < 7.5:
        step = 5.0 * exp
    else:
        step = 10.0 * exp
    start = np.ceil(lo / step) * step
    return np.arange(start, hi + step * 0.0001, step)


def _nice_longitude_tick_degrees(lo: float, hi: float, max_ticks: int = 14) -> np.ndarray:
    """Longitude ticks preferring multiples/submultiples of 30°; generic fallback.

    Among valid nice steps, uses the finest step that still keeps tick count
    within a small margin above ``max_ticks`` (coarse-first iteration with last
    valid assignment wins).
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return np.array([])
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    preferred = (
        360, 180, 120, 90, 60, 45, 30, 20, 15, 12, 10, 6, 5, 4, 3, 2, 1,
        0.5, 0.25,
    )
    cap = max_ticks + 6
    best: Optional[np.ndarray] = None
    for step in preferred:
        start = np.ceil(lo / step) * step
        ticks = np.arange(start, hi + step * 0.0001, step)
        if 1 <= len(ticks) <= cap:
            best = ticks
    for n in range(1, 31):
        step = 30.0 / n
        if step < 0.15:
            break
        start = np.ceil(lo / step) * step
        ticks = np.arange(start, hi + step * 0.0001, step)
        if 1 <= len(ticks) <= cap:
            best = ticks
    if best is not None:
        return best
    return _nice_tick_values(lo, hi, max_ticks)


class TiledImageWidget(QAbstractScrollArea):
    """Scroll area that renders a large image in tiles.

    Only the visible viewport region is rendered on each paint, making
    arbitrary zoom levels efficient without allocating oversized QPixmaps.
    Supports independent X and Y zoom factors.
    """

    # ------------------------------------------------------------------ #
    #  Signals                                                             #
    # ------------------------------------------------------------------ #

    # (pixel_x, pixel_y, in_bounds)
    mouse_moved = pyqtSignal(float, float, bool)
    # (x_zoom, y_zoom)
    zoom_changed = pyqtSignal(float, float)
    # right-click: (pixel_x, pixel_y)
    right_clicked = pyqtSignal(float, float)
    # Ctrl+left-click: (pixel_x, pixel_y)
    ctrl_clicked = pyqtSignal(float, float)

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setViewportMargins(0, 0, 0, 0)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Image data (array: row 0 = inner, displayed flipped)
        self._image_ma: Optional[ma.MaskedArray] = None
        self._n_radii: int = 0
        self._n_long: int = 0
        self._long_interval: float = 0.02
        self._radial_interval: float = 5.0

        # Stretch
        self._black: float = 0.0
        self._white: float = 1.0
        self._gamma: float = 0.5

        # Zoom
        self._x_zoom: float = 1.0
        self._y_zoom: float = 1.0

        # Color-by: (n_long, 3) float32 in [0,1], or None for greyscale
        self._color_column: Optional[np.ndarray] = None

        # Show-radii: display pixel_y rows to highlight in green
        self._show_radii_pixel_ys: list[int] = []

        # Optional FOV axis annotations (mean core → Y tick labels as offset km)
        self._show_lon_ticks: bool = False
        self._show_rad_ticks: bool = False
        self._mean_core_km: float = 0.0
        # Max corot longitude (deg) for ticks/info; None → n_long * long_interval
        self._longitude_extent_hi_deg: Optional[float] = None

        # Pan state
        self._drag_start_global: Optional[QPoint] = None
        self._drag_start_scroll: tuple[int, int] = (0, 0)

        # Zoom-to-area (Shift+drag)
        self._rubber_band: Optional[QRubberBand] = None
        self._rubber_origin: Optional[QPoint] = None

        # Keep numpy array alive while QImage uses its buffer
        self._last_rgb: Optional[np.ndarray] = None

        self.horizontalScrollBar().valueChanged.connect(
            lambda _: self.viewport().update())
        self.verticalScrollBar().valueChanged.connect(
            lambda _: self.viewport().update())

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_image(
        self,
        image_ma: ma.MaskedArray,
        long_interval: float,
        radial_interval: float,
        longitude_extent_hi_deg: Optional[float] = None,
    ) -> None:
        """Load new image data and reset scroll to origin.

        ``longitude_extent_hi_deg`` caps corot longitude (e.g. ``360 - interval``
        for full-ring reproj grids per ``display_reproj_img.py``). If None, uses
        ``n_long * long_interval``.
        """
        if image_ma.ndim != 2:
            raise ValueError(
                f'image_ma must be 2-D, got ndim={image_ma.ndim}, shape={image_ma.shape}')
        self._image_ma = image_ma
        self._n_radii, self._n_long = image_ma.shape
        self._long_interval = long_interval
        self._radial_interval = radial_interval
        self._longitude_extent_hi_deg = longitude_extent_hi_deg
        self._x_zoom = 1.0
        self._y_zoom = 1.0
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)
        self._update_scroll_range()
        self.viewport().update()

    def _longitude_axis_max_deg(self) -> float:
        if self._longitude_extent_hi_deg is not None:
            return float(self._longitude_extent_hi_deg)
        if self._n_long > 0:
            return float(self._n_long * self._long_interval)
        return 360.0

    def set_stretch(self, black: float, white: float, gamma: float) -> None:
        self._black = black
        self._white = max(white, black + 1e-10)
        self._gamma = max(gamma, 0.01)
        self.viewport().update()

    def set_zoom(
        self,
        x_zoom: float,
        y_zoom: float,
        anchor_vx: Optional[int] = None,
        anchor_vy: Optional[int] = None,
        anchor_img_x: Optional[float] = None,
        anchor_img_y: Optional[float] = None,
    ) -> None:
        """Set zoom, optionally anchoring a viewport point to an image coord."""
        self._apply_zoom(x_zoom, y_zoom, anchor_vx, anchor_vy,
                         anchor_img_x, anchor_img_y)

    def get_zoom(self) -> tuple[float, float]:
        return self._x_zoom, self._y_zoom

    def longitude_fov_span_px(self) -> int:
        """Pixel width for corotating-longitude span (EW xlim / mosaic lon ticks).

        The vertical scrollbar sits beside the viewport and narrows it, but the
        mosaic column width is viewport + scrollbar. Longitude scale should use
        that full width so the EW plot matches the mosaic (same virtual width as
        when no radial scrollbar is shown).
        """
        vp = max(1, self.viewport().width())
        vb = self.verticalScrollBar()
        if vb.isVisible():
            return max(1, vp + vb.width())
        return vp

    def get_min_zoom(self) -> tuple[float, float]:
        """Minimum X/Y zoom so virtual image size fills the viewport (no undershoot)."""
        return self._min_zoom_xy()

    def _min_zoom_xy(self) -> tuple[float, float]:
        if self._image_ma is None or self._n_long < 1 or self._n_radii < 1:
            return (0.05, 0.05)
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        return (float(vw) / float(self._n_long), float(vh) / float(self._n_radii))

    def set_color_column(self, color_column: Optional[np.ndarray]) -> None:
        """Set per-column RGB tinting used by ``_do_paint``.

        ``None`` disables tinting. Otherwise ``color_column`` must be a
        ``numpy.ndarray`` with ``ndim == 2``, ``shape[1] == 3``, and first
        dimension matching ``self._n_long`` when an image is loaded
        (``self._n_long > 0``). Values are coerced to float and must lie in
        [0, 1].
        """
        if color_column is None:
            self._color_column = None
            self.viewport().update()
            return
        if not isinstance(color_column, np.ndarray):
            raise ValueError(
                f'color_column must be None or numpy.ndarray, got {type(color_column)}')
        if color_column.ndim != 2 or color_column.shape[1] != 3:
            raise ValueError(
                f'color_column must have shape (n_long, 3), got shape {color_column.shape}')
        if self._n_long > 0 and color_column.shape[0] != self._n_long:
            raise ValueError(
                f'color_column length {color_column.shape[0]} does not match '
                f'n_long={self._n_long} (expected by _do_paint)')
        cc = np.asarray(color_column, dtype=np.float64)
        if np.any(cc < 0.0) or np.any(cc > 1.0):
            raise ValueError('color_column values must lie in [0, 1]')
        self._color_column = cc.astype(np.float32)
        self.viewport().update()

    def set_show_radii(self, pixel_ys: list[int]) -> None:
        """Set display pixel_y rows to draw as green horizontal lines."""
        self._show_radii_pixel_ys = pixel_ys
        self.viewport().update()

    def set_axis_tick_options(
        self,
        show_longitude: bool,
        show_radius: bool,
        mean_core_km: float,
    ) -> None:
        """Toggle FOV tick overlays; mean core defines radius offset = 0 on Y."""
        self._show_lon_ticks = show_longitude
        self._show_rad_ticks = show_radius
        self._mean_core_km = float(mean_core_km)
        self.viewport().update()

    def viewport_to_pixel(self, vx: int, vy: int) -> tuple[float, float]:
        """Convert viewport screen coords to image pixel coords."""
        hv = self.horizontalScrollBar().value()
        vv = self.verticalScrollBar().value()
        return (hv + vx) / self._x_zoom, (vv + vy) / self._y_zoom

    def pixel_to_physical(
        self, pixel_x: float, pixel_y: float
    ) -> tuple[float, float]:
        """Return (corot_long_deg, rel_radius_km) from image pixel coords."""
        corot = float(pixel_x * self._long_interval)
        corot = min(corot, self._longitude_axis_max_deg())
        rel_r = ((self._n_radii - 1) / 2.0 - pixel_y) * self._radial_interval
        return corot, rel_r

    def pixel_y_to_arr_row(self, pixel_y: float) -> int:
        """Convert display pixel_y (0=outer) to array row index (0=inner).

        ``pixel_y`` is clamped to [0, n_radii - 1] before conversion.
        """
        if self._n_radii < 1:
            return 0
        cy = float(np.clip(pixel_y, 0.0, float(self._n_radii - 1)))
        return (self._n_radii - 1) - int(cy)

    def scroll_to_pixel(self, pixel_x: float, pixel_y: float) -> None:
        """Scroll so that the given image pixel is centred in the viewport."""
        vw = self.viewport().width()
        vh = self.viewport().height()
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        h = int(pixel_x * self._x_zoom - vw / 2)
        v = int(pixel_y * self._y_zoom - vh / 2)
        hbar.setValue(max(0, min(hbar.maximum(), h)))
        vbar.setValue(max(0, min(vbar.maximum(), v)))

    def render_viewport_to_image(self) -> QImage:
        """Render the currently-visible viewport to a QImage (for Save FOV)."""
        vw = self.viewport().width()
        vh = self.viewport().height()
        img = QImage(vw, vh, QImage.Format.Format_RGB888)
        painter = QPainter(img)
        self._do_paint(painter, vw, vh)
        painter.end()
        return img

    # ------------------------------------------------------------------ #
    #  Qt overrides                                                        #
    # ------------------------------------------------------------------ #

    def viewportEvent(self, event: QEvent) -> bool:
        t = event.type()
        if t == QEvent.Type.Paint:
            painter = QPainter(self.viewport())
            self._do_paint(painter,
                           self.viewport().width(),
                           self.viewport().height())
            painter.end()
            return True
        if t == QEvent.Type.MouseButtonPress:
            self._mouse_press(event)
            return True
        if t == QEvent.Type.MouseMove:
            self._mouse_move(event)
            return True
        if t == QEvent.Type.MouseButtonRelease:
            self._mouse_release(event)
            return True
        if t == QEvent.Type.MouseButtonDblClick:
            return True
        return super().viewportEvent(event)

    def wheelEvent(self, event) -> None:
        """Zoom at cursor: both axes, or X-only with Shift, or Y-only with Ctrl."""
        if self._image_ma is None:
            event.accept()
            return
        pos = event.position().toPoint()
        vp_pos = self.viewport().mapFromParent(pos)
        vx, vy = vp_pos.x(), vp_pos.y()
        factor = 1.2 if event.angleDelta().y() > 0 else (1.0 / 1.2)
        img_x, img_y = self.viewport_to_pixel(vx, vy)
        min_x, min_y = self._min_zoom_xy()
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if shift and not ctrl:
            new_xz = float(np.clip(self._x_zoom * factor, min_x, 100.0))
            new_yz = self._y_zoom
        elif ctrl and not shift:
            new_xz = self._x_zoom
            new_yz = float(np.clip(self._y_zoom * factor, min_y, 100.0))
        else:
            new_xz = float(np.clip(self._x_zoom * factor, min_x, 100.0))
            new_yz = float(np.clip(self._y_zoom * factor, min_y, 100.0))
        self._apply_zoom(new_xz, new_yz, vx, vy, img_x, img_y)
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scroll_range()
        if self._image_ma is not None:
            min_x, min_y = self._min_zoom_xy()
            if self._x_zoom < min_x - 1e-12 or self._y_zoom < min_y - 1e-12:
                self._apply_zoom(
                    max(self._x_zoom, min_x), max(self._y_zoom, min_y),
                    None, None, None, None)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _update_scroll_range(self) -> None:
        if self._image_ma is None:
            return
        vw = self.viewport().width()
        vh = self.viewport().height()
        virtual_w = max(1, int(self._n_long * self._x_zoom))
        virtual_h = max(1, int(self._n_radii * self._y_zoom))
        hbar = self.horizontalScrollBar()
        hbar.setRange(0, max(0, virtual_w - vw))
        hbar.setPageStep(vw)
        vbar = self.verticalScrollBar()
        vbar.setRange(0, max(0, virtual_h - vh))
        vbar.setPageStep(vh)

    def _apply_zoom(
        self,
        new_xz: float,
        new_yz: float,
        anchor_vx: Optional[int],
        anchor_vy: Optional[int],
        anchor_img_x: Optional[float],
        anchor_img_y: Optional[float],
    ) -> None:
        min_x, min_y = self._min_zoom_xy()
        new_xz = float(np.clip(new_xz, min_x, 100.0))
        new_yz = float(np.clip(new_yz, min_y, 100.0))
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        vw = self.viewport().width()
        vh = self.viewport().height()

        if anchor_vx is not None and anchor_img_x is not None:
            new_hv = int(anchor_img_x * new_xz - anchor_vx)
        else:
            cx = (hbar.value() + vw / 2) / self._x_zoom
            new_hv = int(cx * new_xz - vw / 2)

        if anchor_vy is not None and anchor_img_y is not None:
            new_vv = int(anchor_img_y * new_yz - anchor_vy)
        else:
            cy = (vbar.value() + vh / 2) / self._y_zoom
            new_vv = int(cy * new_yz - vh / 2)

        self._x_zoom = new_xz
        self._y_zoom = new_yz
        self._update_scroll_range()
        hbar.setValue(max(0, min(hbar.maximum(), new_hv)))
        vbar.setValue(max(0, min(vbar.maximum(), new_vv)))
        self.zoom_changed.emit(self._x_zoom, self._y_zoom)
        self.viewport().update()

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _do_paint(self, painter: QPainter, vw: int, vh: int) -> None:
        painter.fillRect(0, 0, vw, vh, Qt.GlobalColor.black)
        if self._image_ma is None or self._n_long == 0 or self._n_radii == 0:
            return

        hv = self.horizontalScrollBar().value()
        vv = self.verticalScrollBar().value()
        xz = self._x_zoom
        yz = self._y_zoom

        # Visible range in image pixel coords
        px_start = max(0, int(np.floor(hv / xz)))
        px_end = min(self._n_long - 1, int(np.ceil((hv + vw) / xz)))
        py_start = max(0, int(np.floor(vv / yz)))
        py_end = min(self._n_radii - 1, int(np.ceil((vv + vh) / yz)))

        if px_start > px_end or py_start > py_end:
            return

        # pixel_y = (n_radii-1) - arr_row, so:
        #   py_start (top of screen, outer) → arr_row = (n_radii-1) - py_start
        #   py_end   (bottom, inner)        → arr_row = (n_radii-1) - py_end
        arr_row_max = min(self._n_radii - 1, (self._n_radii - 1) - py_start)
        arr_row_min = max(0, (self._n_radii - 1) - py_end)

        tile_raw = self._image_ma[arr_row_min:arr_row_max + 1,
                                  px_start:px_end + 1]
        # Flip vertically: row 0 of tile_display = outer = py_start (top)
        tile = tile_raw[::-1, :]

        tile_h, tile_w = tile.shape
        tile_mask = ma.getmaskarray(tile)
        tile_data = np.nan_to_num(tile.filled(0.0), nan=0.0).astype(np.float32)

        # Contrast stretch
        b, w, g = self._black, self._white, self._gamma
        if w <= b:
            w = b + 1e-10
        stretched = ((np.clip(tile_data, b, w) - b) / (w - b)) ** g
        gray = (stretched * 255.0).astype(np.uint8)

        # Build RGB (apply colour-by tinting if active)
        if self._color_column is not None and len(self._color_column) > 0:
            col_idx = np.clip(
                np.arange(px_start, px_end + 1, dtype=np.intp),
                0, len(self._color_column) - 1,
            )
            tint = self._color_column[col_idx].astype(np.float32)  # (w, 3)
            gray_f = gray[:, :, np.newaxis].astype(np.float32)     # (h, w, 1)
            rgb = np.clip(gray_f * tint[np.newaxis, :, :], 0, 255).astype(np.uint8)
        else:
            rgb = np.stack([gray, gray, gray], axis=2)

        # Mask overlay: masked pixels rendered as dark red
        if np.any(tile_mask):
            rgb[tile_mask, 0] = 180
            rgb[tile_mask, 1] = 0
            rgb[tile_mask, 2] = 0

        # Show-radii overlay: green horizontal lines
        for py in self._show_radii_pixel_ys:
            if py_start <= py <= py_end:
                tile_row = py - py_start
                if 0 <= tile_row < tile_h:
                    rgb[tile_row, :, 0] = 0
                    rgb[tile_row, :, 1] = 220
                    rgb[tile_row, :, 2] = 0

        # Screen destination rectangle
        dest_x = int(round(px_start * xz)) - hv
        dest_y = int(round(py_start * yz)) - vv
        dest_w = max(1, int(round((px_end + 1) * xz)) - hv - dest_x)
        dest_h = max(1, int(round((py_end + 1) * yz)) - vv - dest_y)

        # Keep numpy array alive while QImage reads its buffer
        self._last_rgb = np.ascontiguousarray(rgb)
        qimg = QImage(
            self._last_rgb.data,
            tile_w, tile_h,
            3 * tile_w,
            QImage.Format.Format_RGB888,
        )
        scaled = qimg.scaled(
            dest_w, dest_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter.drawImage(dest_x, dest_y, scaled)

        if self._show_lon_ticks or self._show_rad_ticks:
            self._draw_fov_tick_overlays(
                painter, vw, vh, hv, vv, xz, yz)

    def _draw_fov_tick_overlays(
        self,
        painter: QPainter,
        vw: int,
        vh: int,
        hv: int,
        vv: int,
        xz: float,
        yz: float,
    ) -> None:
        """Draw longitude (bottom) and/or radius (left) ticks aligned with FOV.

        Longitude and radius are drawn independently (may overlap). Tick marks
        use exact rounded screen positions; labels are inset from viewport
        edges, shifting beside the tick when centering would violate margins.
        """
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        pen = QPen(QColor(220, 220, 220))
        painter.setPen(pen)
        bg_col = QColor(25, 25, 28, 210)
        pad = 3
        EDGE = 8

        _BOTTOM = 30
        _LEFT = 52
        _TICK_X = _LEFT - 4

        vp_w = vw  # drawable viewport width (longitude ticks clip here)

        def _lon_label_left(ix: int, tw: int) -> int:
            ideal = ix - tw // 2
            if ideal < EDGE:
                return int(min(ix + 4, max(EDGE, vp_w - EDGE - tw)))
            if ideal + tw > vp_w - EDGE:
                return int(max(ix - tw - 4, EDGE))
            return int(ideal)

        # --- Longitude (bottom): no coupling to radius ---
        if self._show_lon_ticks:
            font_x = QFont()
            font_x.setPointSize(10)
            painter.setFont(font_x)
            fm_x = painter.fontMetrics()
            vw_lon = self.longitude_fov_span_px()
            px0 = hv / xz
            px1 = (hv + vw_lon) / xz
            hi = self._longitude_axis_max_deg()
            c0 = float(np.clip(px0 * self._long_interval, 0.0, hi))
            c1 = float(np.clip(px1 * self._long_interval, 0.0, hi))
            tick_y0 = vh - _BOTTOM
            tick_y1 = vh - EDGE
            text_baseline = vh - EDGE
            th = fm_x.ascent() + fm_x.descent()
            tty = text_baseline - fm_x.ascent()

            for cor in _nice_longitude_tick_degrees(c0, c1, 14):
                img_x = cor / self._long_interval
                sx = float(img_x * xz - hv)
                if not (-20 < sx < vp_w + 20):
                    continue
                ix = int(round(sx))
                if not (0 <= ix < vp_w):
                    continue
                txt = f'{cor:.0f}°'
                tw = fm_x.horizontalAdvance(txt)
                if tw >= vp_w - 2 * EDGE:
                    continue
                tx = _lon_label_left(ix, tw)
                tx = int(np.clip(tx, EDGE, vp_w - EDGE - tw))
                tick_bar = QRect(ix, tick_y0, 1, max(1, tick_y1 - tick_y0))
                text_r = QRect(tx, tty, tw, th)
                bg_lon = tick_bar.united(text_r).adjusted(-pad, -pad, pad, pad)
                painter.fillRect(bg_lon, bg_col)
                painter.drawLine(ix, tick_y0, ix, tick_y1)
                painter.drawText(tx, text_baseline, txt)

        # --- Radius (left): no coupling to longitude ---
        if self._show_rad_ticks:
            font_y = QFont()
            font_y.setPointSize(10)
            painter.setFont(font_y)
            label_h = 22
            half_h = label_h // 2
            text_left = EDGE
            text_w = max(1, _TICK_X - 4 - text_left)

            py0 = vv / yz
            py1 = (vv + vh) / yz
            r0 = self._pixel_y_to_abs_radius_km(py1)
            r1 = self._pixel_y_to_abs_radius_km(py0)
            lo, hi = min(r0, r1), max(r0, r1)
            mc = self._mean_core_km
            off_lo, off_hi = lo - mc, hi - mc
            for off_km in _nice_tick_values(off_lo, off_hi, 8):
                abs_km = off_km + mc
                sy = self._abs_radius_km_to_screen_y(abs_km, yz, vv)
                if not (-20 < sy < vh + 20):
                    continue
                iy = int(round(sy))
                if not (0 <= iy < vh):
                    continue
                txt = f'{off_km:.0f}'
                tr = QRect(text_left, iy - half_h, text_w, label_h)
                if tr.top() < EDGE:
                    tr.moveTop(EDGE)
                if tr.bottom() > vh - EDGE:
                    tr.moveBottom(vh - EDGE)
                line_r = QRect(_TICK_X, iy, max(1, _LEFT - _TICK_X), 1)
                bg_r = line_r.united(tr).adjusted(-pad, -pad, pad, pad)
                painter.fillRect(bg_r, bg_col)
                painter.drawLine(_TICK_X, iy, _LEFT, iy)
                painter.drawText(
                    tr,
                    int(Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter),
                    txt)

    def _pixel_y_to_abs_radius_km(self, pixel_y: float) -> float:
        rel = ((self._n_radii - 1) / 2.0 - pixel_y) * self._radial_interval
        return rel + self._mean_core_km

    def _abs_radius_km_to_screen_y(
        self, abs_km: float, yz: float, vv: int
    ) -> float:
        rel = abs_km - self._mean_core_km
        pixel_y = (self._n_radii - 1) / 2.0 - rel / self._radial_interval
        return pixel_y * yz - vv

    # ------------------------------------------------------------------ #
    #  Mouse events                                                        #
    # ------------------------------------------------------------------ #

    def _mouse_press(self, event) -> None:
        btn = event.button()
        mods = event.modifiers()
        vx = int(event.position().x())
        vy = int(event.position().y())
        px, py = self.viewport_to_pixel(vx, vy)
        in_bounds = (self._image_ma is not None
                     and 0 <= px < self._n_long
                     and 0 <= py < self._n_radii)

        if btn == Qt.MouseButton.RightButton:
            if in_bounds:
                self.right_clicked.emit(px, py)
            return

        if btn == Qt.MouseButton.LeftButton:
            if mods & Qt.KeyboardModifier.ControlModifier:
                if in_bounds:
                    self.ctrl_clicked.emit(px, py)
                return

            if mods & Qt.KeyboardModifier.ShiftModifier:
                # Start zoom-to-area rubber band
                origin = QPoint(vx, vy)
                self._rubber_origin = origin
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(
                        QRubberBand.Shape.Rectangle, self.viewport())
                self._rubber_band.setGeometry(QRect(origin, QSize()))
                self._rubber_band.show()
                return

            # Normal left-drag: pan
            self._drag_start_global = event.globalPosition().toPoint()
            self._drag_start_scroll = (
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _mouse_move(self, event) -> None:
        vx = int(event.position().x())
        vy = int(event.position().y())
        px, py = self.viewport_to_pixel(vx, vy)
        in_bounds = (self._image_ma is not None
                     and 0 <= px < self._n_long
                     and 0 <= py < self._n_radii)
        self.mouse_moved.emit(px, py, in_bounds)

        mods = event.modifiers()

        # Rubber-band update (Shift+drag)
        if (self._rubber_origin is not None
                and self._rubber_band is not None
                and mods & Qt.KeyboardModifier.ShiftModifier):
            self._rubber_band.setGeometry(
                QRect(self._rubber_origin, QPoint(vx, vy)).normalized())
            return

        # Pan drag
        if self._drag_start_global is not None:
            delta = event.globalPosition().toPoint() - self._drag_start_global
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(int(np.clip(
                self._drag_start_scroll[0] - delta.x(), 0, hbar.maximum())))
            vbar.setValue(int(np.clip(
                self._drag_start_scroll[1] - delta.y(), 0, vbar.maximum())))

    def _mouse_release(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if (self._rubber_band is not None
                and self._rubber_band.isVisible()):
            rect = self._rubber_band.geometry()
            self._rubber_band.hide()
            self._rubber_origin = None
            self._apply_zoom_to_rect(rect)
            return
        self._drag_start_global = None
        self.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _apply_zoom_to_rect(self, viewport_rect: QRect) -> None:
        """Zoom so the rubber-band selection fills the viewport."""
        if (viewport_rect.width() < 4
                or viewport_rect.height() < 4
                or self._image_ma is None):
            return
        vw = self.viewport().width()
        vh = self.viewport().height()
        hv = self.horizontalScrollBar().value()
        vv = self.verticalScrollBar().value()

        # Convert rubber band to image pixel space
        px_l = (hv + viewport_rect.left()) / self._x_zoom
        px_r = (hv + viewport_rect.right()) / self._x_zoom
        py_t = (vv + viewport_rect.top()) / self._y_zoom
        py_b = (vv + viewport_rect.bottom()) / self._y_zoom

        pix_w = max(px_r - px_l, 0.5)
        pix_h = max(py_b - py_t, 0.5)
        min_x, min_y = self._min_zoom_xy()
        new_xz = float(np.clip(vw / pix_w, min_x, 100.0))
        new_yz = float(np.clip(vh / pix_h, min_y, 100.0))

        self._x_zoom = new_xz
        self._y_zoom = new_yz
        self._update_scroll_range()
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        hx = int(np.clip(px_l * new_xz, 0, hbar.maximum()))
        hy = int(np.clip(py_t * new_yz, 0, vbar.maximum()))
        hbar.setValue(hx)
        vbar.setValue(hy)
        self.zoom_changed.emit(self._x_zoom, self._y_zoom)
        self.viewport().update()
