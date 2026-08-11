"""MosaicWindow: main window for browsing F ring mosaics."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import numpy as np
import numpy.ma as ma

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QRadioButton,
    QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from catalog import FilterCriteria, MosaicCatalog, MosaicRecord
from filter_dialog import FilterDialog
from image_renderer import TiledImageWidget, slider_to_zoom, zoom_to_slider
from pds4_reader import (
    compute_default_stretch, get_element,
    get_mosaic_name_from_mosaic_label, get_mosaic_name_from_reproj_img_label,
    lidvid_to_reproj_name, read_mosaic_ma, read_reproj_img_ma,
    reproj_product_stem_from_label,
)
from utils import (
    build_full_width_metadata, compute_color_column, compute_ew,
    compute_ewmu, show_radii_to_pixel_ys, tdb_to_utc_str,
)

logger = logging.getLogger(__name__)

# ``compute_default_stretch`` white-point ignore = 1 − percentile (e.g. 0.02 → ~98%).
_STRETCH_BRIGHT_WHITE_IGNORE_FRAC = 0.02

# User-defined EW radial bands: distinct from full-mosaic curve (steelblue).
_EW_BAND_COLOR_CYCLE = (
    '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
    '#e377c2', '#bcbd22', '#17becf', '#8da0cb', '#ff9896',
)

# Baseline status text (transient messages from showMessage() overlay it).
STATUS_BAR_HINT_MOSAIC = (
    'Mouse wheel zooms both axes (Shift+wheel: X only, Ctrl+wheel: Y only). '
    'Shift+Left to zoom to region, Left drag to pan, '
    'Ctrl+Left to set EW inner and outer radii, Right to open reprojection '
    'image.')
STATUS_BAR_HINT_REPROJ = (
    'Mouse wheel zooms both axes (Shift+wheel: X only, Ctrl+wheel: Y only). '
    'Shift+Left to zoom to region, Left drag to pan, '
    'Ctrl+Left to set EW inner and outer radii.')

# Color-by: meta field and optional fixed (lo, hi) for ``compute_color_column``.
_COLORBY_REL_META_FIELD: dict[str, str] = {
    'rel_rad_res': 'rings_radial_resolution',
    'rel_ang_res': 'rings_longitudinal_resolution',
    'rel_phase': 'rings_phase_angle',
    'rel_emission': 'rings_emission_angle',
    'rel_inertial': 'rings_inertial_ring_longitude',
    'rel_true_anomaly': 'true_anomaly',
}
_COLORBY_ABS_RANGE: dict[str, tuple[str, float, float]] = {
    'abs_phase': ('rings_phase_angle', 0.0, 180.0),
    'abs_emission': ('rings_emission_angle', 0.0, 180.0),
    'abs_inertial': ('rings_inertial_ring_longitude', 0.0, 360.0),
    'abs_true_anomaly': ('true_anomaly', 0.0, 360.0),
}

# ======================================================================= #
#  Loaded mosaic data                                                      #
# ======================================================================= #

@dataclass
class MosaicData:
    """All data for a single loaded mosaic."""
    label: Any
    image_ma: ma.MaskedArray      # (n_radii, n_long); row 0=inner
    meta: dict                    # full-width masked arrays per field
    image_table: dict             # image_index int -> LIDVID str
    obsid: str                    # e.g. iss_029rf_fmovie001_vims
    # Parent directory name of the loaded .lblx (matches data_reproj_img/<this>/).
    # Differs from ``obsid`` for browse products whose LID contains ``_browse_mosaic``.
    bundle_host_dir: str
    long_interval: float
    radial_interval: float
    n_radii: int
    n_long: int
    mean_core_radius: float
    black: float                  # auto-computed default black point
    white: float                  # auto-computed default white point
    ew: ma.MaskedArray            # (n_long,) equivalent width
    ew_mu: ma.MaskedArray         # (n_long,) EW * |cos(emission)|
    ew_mean: float
    ew_std: float
    ewmu_mean: float
    ewmu_std: float
    image_vmin: float               # min I/F over valid (unmasked) pixels
    image_vmax: float               # max I/F over valid pixels
    is_reproj: bool = False
    reproj_name: str = ''
    longitude_extent_hi_deg: Optional[float] = None


def _mean_std_masked_1d(arr: ma.MaskedArray) -> tuple[float, float]:
    valid = arr.compressed()
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.mean(valid)), float(np.std(valid))


def _image_vmin_vmax(image_ma: ma.MaskedArray) -> tuple[float, float]:
    valid_img = image_ma.compressed()
    if valid_img.size > 0:
        return float(np.min(valid_img)), float(np.max(valid_img))
    return 0.0, 1.0


def _bundle_host_dir_from_label_path(label_path: str) -> str:
    """Directory name containing the product label (bundle-relative host folder)."""
    return os.path.basename(os.path.dirname(os.path.abspath(label_path)))


def load_mosaic(label_path: str) -> MosaicData:
    """Load a mosaic from its .lblx path and return a MosaicData instance."""
    label, image_ma, meta_params, img_table = read_mosaic_ma(
        label_path, include_image_table=True)

    long_interval = get_element(
        label, 'rings:reprojection_grid_longitudinal_sampling_interval')
    radial_interval = get_element(
        label, 'rings:reprojection_grid_radial_sampling_interval')
    n_radii, n_long = image_ma.shape
    obsid = get_mosaic_name_from_mosaic_label(label)
    bundle_host_dir = _bundle_host_dir_from_label_path(label_path)

    meta = build_full_width_metadata(meta_params, n_long, long_interval)

    mean_core = float(np.mean(meta['core_radius'].compressed()))

    black, white = compute_default_stretch(image_ma)

    ew = compute_ew(image_ma, radial_interval)
    emission_deg = meta['rings_emission_angle']
    ew_mu = compute_ewmu(ew, emission_deg)

    ew_mean, ew_std = _mean_std_masked_1d(ew)
    ewmu_mean, ewmu_std = _mean_std_masked_1d(ew_mu)
    image_vmin, image_vmax = _image_vmin_vmax(image_ma)

    return MosaicData(
        label=label,
        image_ma=image_ma,
        meta=meta,
        image_table=img_table,
        obsid=obsid,
        bundle_host_dir=bundle_host_dir,
        long_interval=long_interval,
        radial_interval=radial_interval,
        n_radii=n_radii,
        n_long=n_long,
        mean_core_radius=mean_core,
        black=black,
        white=white,
        ew=ew,
        ew_mu=ew_mu,
        ew_mean=ew_mean,
        ew_std=ew_std,
        ewmu_mean=ewmu_mean,
        ewmu_std=ewmu_std,
        image_vmin=image_vmin,
        image_vmax=image_vmax,
    )


def load_reproj(label_path: str) -> MosaicData:
    """Load a reprojected image (full 360° grid) from its .lblx path."""
    label, image_ma, meta_params = read_reproj_img_ma(label_path)
    long_interval = get_element(
        label, 'rings:reprojection_grid_longitudinal_sampling_interval')
    radial_interval = get_element(
        label, 'rings:reprojection_grid_radial_sampling_interval')
    n_radii, n_long = image_ma.shape
    obsid = get_mosaic_name_from_reproj_img_label(label)
    bundle_host_dir = _bundle_host_dir_from_label_path(label_path)
    reproj_name = reproj_product_stem_from_label(label)
    meta = build_full_width_metadata(meta_params, n_long, long_interval)
    mean_core = float(np.mean(meta['core_radius'].compressed()))

    black, white = compute_default_stretch(image_ma)
    ew = compute_ew(image_ma, radial_interval)
    emission_deg = meta['rings_emission_angle']
    ew_mu = compute_ewmu(ew, emission_deg)

    ew_mean, ew_std = _mean_std_masked_1d(ew)
    ewmu_mean, ewmu_std = _mean_std_masked_1d(ew_mu)
    image_vmin, image_vmax = _image_vmin_vmax(image_ma)

    # Open longitude interval at 360° (matches display_reproj_img.py imshow extent).
    lon_hi = 360.0 - float(long_interval)

    return MosaicData(
        label=label,
        image_ma=image_ma,
        meta=meta,
        image_table={},
        obsid=obsid,
        bundle_host_dir=bundle_host_dir,
        long_interval=long_interval,
        radial_interval=radial_interval,
        n_radii=n_radii,
        n_long=n_long,
        mean_core_radius=mean_core,
        black=black,
        white=white,
        ew=ew,
        ew_mu=ew_mu,
        ew_mean=ew_mean,
        ew_std=ew_std,
        ewmu_mean=ewmu_mean,
        ewmu_std=ewmu_std,
        image_vmin=image_vmin,
        image_vmax=image_vmax,
        is_reproj=True,
        reproj_name=reproj_name,
        longitude_extent_hi_deg=lon_hi,
    )


# ======================================================================= #
#  Sync widget helper                                                       #
# ======================================================================= #

class _SyncedSlider:
    """Keeps a QLineEdit and QSlider in sync."""

    def __init__(
        self,
        line_edit: QLineEdit,
        slider: QSlider,
        lo: float,
        hi: float,
        fmt: str = '%.4f',
        on_change=None,
    ) -> None:
        self._le = line_edit
        self._sl = slider
        self._lo = lo
        self._hi = hi
        self._fmt = fmt
        self._on_change = on_change
        self._updating = False

        self._sl.valueChanged.connect(self._slider_moved)
        self._le.editingFinished.connect(self._edit_done)

    def _to_slider(self, val: float) -> int:
        if self._hi <= self._lo:
            return 0
        pos = (val - self._lo) / (self._hi - self._lo) * 1000.0
        return int(round(np.clip(pos, 0, 1000)))

    def _from_slider(self, pos: int) -> float:
        return self._lo + (self._hi - self._lo) * pos / 1000.0

    def _slider_moved(self, pos: int) -> None:
        if self._updating:
            return
        val = self._from_slider(pos)
        self._updating = True
        self._le.setText(self._fmt % val)
        self._updating = False
        if self._on_change:
            self._on_change(val)

    def _edit_done(self) -> None:
        if self._updating:
            return
        try:
            val = float(self._le.text())
        except ValueError:
            return
        val = max(self._lo, min(self._hi, val))
        self._updating = True
        self._sl.setValue(self._to_slider(val))
        self._le.setText(self._fmt % val)
        self._updating = False
        if self._on_change:
            self._on_change(val)

    def set_range(self, lo: float, hi: float) -> None:
        self._lo = lo
        self._hi = hi

    def set_value(self, val: float) -> None:
        self._updating = True
        self._sl.setValue(self._to_slider(val))
        self._le.setText(self._fmt % val)
        self._updating = False

    def get_value(self) -> float:
        try:
            return float(self._le.text())
        except ValueError:
            return self._from_slider(self._sl.value())


# ======================================================================= #
#  MosaicWindow                                                            #
# ======================================================================= #

# Pixels reserved left of mosaic to align image with EW Profile axes (y-axis labels).
EW_PROFILE_LEFT_GUTTER_PX = 58


class MosaicWindow(QMainWindow):
    """One independent mosaic viewer window."""

    # Extra top-level windows (reproj / New Win) must stay referenced; otherwise
    # Python's GC can collect them after the slot returns while Qt still shows them.
    _retained_top_level_windows: ClassVar[list[QMainWindow]] = []

    @staticmethod
    def _retain_top_level_window(win: QMainWindow) -> None:
        MosaicWindow._retained_top_level_windows.append(win)

        def _release(*_args) -> None:
            try:
                MosaicWindow._retained_top_level_windows.remove(win)
            except ValueError:
                pass

        win.destroyed.connect(_release)

    def __init__(
        self,
        catalog: Optional[MosaicCatalog] = None,
        bundle_path: str = '',
        show_radii: Optional[list[float]] = None,  # km relative to mean core
        criteria: Optional[FilterCriteria] = None,
        initial_record: Optional[MosaicRecord] = None,
        initial_black: Optional[float] = None,
        initial_white: Optional[float] = None,
        initial_gamma: float = 0.5,
        reproj_label_path: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._reproj_mode = reproj_label_path is not None
        if not self._reproj_mode and catalog is None:
            raise TypeError('catalog is required unless reproj_label_path is set')
        self._catalog = catalog
        self._bundle_path = bundle_path
        self._reproj_label_path = reproj_label_path
        self._colorby_include_image_number = not self._reproj_mode
        self._show_radii_rel_km = show_radii or []
        self._criteria = (criteria or FilterCriteria()).copy()
        self._filtered: list[MosaicRecord] = []
        self._current_idx: int = 0
        self._mosaic_data: Optional[MosaicData] = None

        # Stretch state
        self._default_black: Optional[float] = initial_black
        self._default_white: Optional[float] = initial_white
        self._default_gamma: float = initial_gamma

        # EW selection state
        self._ew_phase: int = 0
        self._ew_first_py: float = 0.0
        # Remembered radial band ranges (array row indices) for corotating EW plot
        self._ew_radial_ranges: list[tuple[int, int]] = []
        self._pending_fit: bool = False
        self._pending_reproj_fit: bool = False
        self._reproj_open_fit_pending: bool = False
        self._is_loading: bool = False
        # Radial profile: reused Line2D (avoid ax.clear() every mouse move).
        self._radial_profile_line: Any = None

        self._setup_ui()
        if self._reproj_mode:
            self.setWindowTitle('Reprojected image — loading…')
            self._load_reproj_label(reproj_label_path)
        else:
            self._update_filtered_list(initial_record)

    def eventFilter(self, obj, event) -> bool:
        if obj is getattr(self, '_ew_canvas', None) and event.type() == QEvent.Type.Resize:
            self._sync_ew_figure_margins()
        return super().eventFilter(obj, event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._reproj_open_fit_pending:
            self._reproj_open_fit_pending = False
            QTimer.singleShot(0, self._deferred_reproj_zoom_fit)

    def _deferred_reproj_zoom_fit(self) -> None:
        if self._mosaic_data is None or not self._mosaic_data.is_reproj:
            return
        self._fit_zoom_to_reproj_data()

    def _status_bar_hint(self) -> str:
        """Baseline status text; reproj windows omit right-click (no reproj-from-here)."""
        return (
            STATUS_BAR_HINT_REPROJ if self._reproj_mode else STATUS_BAR_HINT_MOSAIC)

    def _safe_radial_canvas_draw(self) -> None:
        """Synchronous redraw if the canvas Qt wrapper is still alive.

        ``draw_idle`` posts ``QTimer.singleShot(0, _draw_idle)`` per canvas; with
        several top-level windows (reproj / New Win) deferred draws can race
        layout and trigger ``RuntimeError: ... FigureCanvasQTAgg has been
        deleted``.  ``draw()`` renders immediately and avoids that queue.
        """
        c = getattr(self, '_radial_canvas', None)
        if c is None:
            return
        try:
            from PyQt6 import sip
            if sip.isdeleted(c):
                return
        except ImportError as e:
            logger.debug('PyQt6 sip import failed: %s', e)
        except ModuleNotFoundError as e:
            logger.debug('PyQt6 sip module not found: %s', e)
        except AttributeError as e:
            logger.debug('sip.isdeleted unavailable: %s', e)
        try:
            c.draw()
        except RuntimeError as e:
            logger.debug('radial canvas draw failed: %s', e)

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        self.setWindowTitle('Mosaic Viewer')
        self.resize(1400, 900)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # Left column: plot options, vertical splitter (radial | corot EW | mosaic)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        self._image_widget = TiledImageWidget()
        self._image_widget.mouse_moved.connect(self._on_mouse_moved)
        self._image_widget.zoom_changed.connect(self._on_zoom_changed)
        self._image_widget.zoom_changed.connect(self._sync_ew_xlim_from_mosaic)
        self._image_widget.horizontalScrollBar().valueChanged.connect(
            self._sync_ew_xlim_from_mosaic)
        # Radial zoom can show the vertical scrollbar; EW xlim must use the same
        # longitude span as mosaic ticks (viewport + scrollbar, not narrowed vp).
        self._image_widget.verticalScrollBar().rangeChanged.connect(
            self._sync_ew_xlim_from_mosaic)
        self._image_widget.right_clicked.connect(self._on_right_click)
        self._image_widget.ctrl_clicked.connect(self._on_ctrl_click)

        self._build_plot_panels(left_layout)

        # Horizontal splitter: left_widget | optional sidebar (mosaic catalog only)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        if not self._reproj_mode:
            sidebar = self._build_sidebar()
            splitter.addWidget(sidebar)
            splitter.setSizes([1150, 210])
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 0)
        else:
            splitter.setSizes([1400])
            splitter.setStretchFactor(0, 1)
        main_layout.addWidget(splitter, stretch=1)

        ctrl = self._build_control_panel()
        main_layout.addWidget(ctrl)

        self.setCentralWidget(central)
        self._cursor_status_lbl = QLabel('')
        self._cursor_status_lbl.setStyleSheet('font-family: monospace;')
        self.statusBar().addPermanentWidget(self._cursor_status_lbl)
        self.statusBar().showMessage(self._status_bar_hint())

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMinimumWidth(175)
        sidebar.setMaximumWidth(280)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        row1 = QHBoxLayout()
        self._btn_filter = QPushButton('Filter')
        self._btn_new_win = QPushButton('New Win')
        self._btn_filter.clicked.connect(self._on_filter)
        self._btn_new_win.clicked.connect(self._on_new_window)
        row1.addWidget(self._btn_filter)
        row1.addWidget(self._btn_new_win)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._btn_prev = QPushButton('< Prev')
        self._btn_next = QPushButton('Next >')
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        row2.addWidget(self._btn_prev)
        row2.addWidget(self._btn_next)
        layout.addLayout(row2)

        self._mosaic_list = QListWidget()
        self._mosaic_list.itemClicked.connect(self._on_list_item_clicked)
        layout.addWidget(self._mosaic_list, stretch=1)

        return sidebar

    def _build_plot_panels(self, left_layout: QVBoxLayout) -> None:
        """Header checkboxes + vertical splitter: radial profile | corot EW | mosaic."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        header = QWidget()
        hh = QHBoxLayout(header)
        hh.setContentsMargins(4, 2, 4, 0)
        hh.setSpacing(10)

        self._chk_lon_ticks = QCheckBox('Longitude axis ticks')
        self._chk_rad_ticks = QCheckBox('Radius axis ticks')
        self._chk_rad_profile = QCheckBox('Radial Profile')
        self._chk_corot_ew = QCheckBox('EW Profile')
        self._chk_corot_use_ewmu = QCheckBox('Use EW x mu')
        self._chk_lon_ticks.setChecked(False)
        self._chk_rad_ticks.setChecked(False)
        self._chk_rad_profile.setChecked(False)
        self._chk_corot_use_ewmu.setChecked(False)
        self._chk_corot_ew.setChecked(False)

        self._chk_lon_ticks.toggled.connect(self._sync_axis_tick_options)
        self._chk_rad_ticks.toggled.connect(self._sync_axis_tick_options)
        self._chk_rad_profile.toggled.connect(self._on_rad_profile_toggled)
        self._chk_corot_use_ewmu.toggled.connect(self._on_corot_ew_mode_changed)
        self._chk_corot_ew.toggled.connect(self._on_corot_ew_panel_toggled)

        for w in (
            self._chk_lon_ticks,
            self._chk_rad_ticks,
            self._chk_rad_profile,
        ):
            hh.addWidget(w)
        hh.addWidget(self._chk_corot_ew)
        self._btn_clear_ew_profile = QPushButton('Clear EW Profile')
        self._btn_clear_ew_profile.clicked.connect(self._on_clear_ew_profile)
        hh.addWidget(self._btn_clear_ew_profile)
        hh.addStretch()
        hh.addWidget(self._chk_corot_use_ewmu)
        left_layout.addWidget(header)

        # --- Radial-at-longitude I/F profile (top plot) ---
        self._rad_wrap = QWidget()
        rad_l = QVBoxLayout(self._rad_wrap)
        rad_l.setContentsMargins(0, 0, 0, 0)
        self._radial_fig = Figure(figsize=(12, 2.8), constrained_layout=True)
        self._radial_ax = self._radial_fig.add_subplot(111)
        self._radial_canvas = FigureCanvasQTAgg(self._radial_fig)
        self._radial_canvas.setMinimumHeight(120)
        self._radial_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        rad_l.addWidget(self._radial_canvas)
        self._init_radial_axes()
        self._rad_wrap.setVisible(False)

        # --- Corotating longitude EW plot ---
        self._cor_wrap = QWidget()
        cor_l = QVBoxLayout(self._cor_wrap)
        cor_l.setContentsMargins(0, 0, 0, 0)
        self._ew_fig = Figure(figsize=(12, 3.6))
        self._ew_ax = self._ew_fig.add_subplot(111)
        self._ew_canvas = FigureCanvasQTAgg(self._ew_fig)
        self._ew_canvas.setMinimumHeight(180)
        self._ew_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._ew_canvas.installEventFilter(self)
        cor_l.addWidget(self._ew_canvas)
        self._init_corot_ew_axes()
        self._cor_wrap.setVisible(False)

        self._plot_splitter = QSplitter(Qt.Orientation.Vertical)
        self._plot_splitter.setChildrenCollapsible(True)
        self._plot_splitter.addWidget(self._rad_wrap)
        self._plot_splitter.addWidget(self._cor_wrap)
        self._plot_splitter.addWidget(self._image_widget)
        self._plot_splitter.setStretchFactor(0, 0)
        self._plot_splitter.setStretchFactor(1, 0)
        self._plot_splitter.setStretchFactor(2, 1)
        self._plot_splitter.setSizes([0, 0, 600])

        left_layout.addWidget(self._plot_splitter, stretch=1)

    def _ew_align_gutter_px(self) -> int:
        return (
            EW_PROFILE_LEFT_GUTTER_PX
            if getattr(self, '_cor_wrap', None) is not None
            and self._cor_wrap.isVisible()
            else 0
        )

    def _sync_ew_mosaic_layout(self) -> None:
        """Inset mosaic viewport when EW Profile is shown so axes align with matplotlib."""
        g = self._ew_align_gutter_px()
        self._image_widget.setViewportMargins(g, 0, 0, 0)
        self._sync_ew_figure_margins()

    def _sync_ew_figure_margins(self) -> None:
        """Match EW axes box left edge to mosaic drawable area (pixel-aligned)."""
        if getattr(self, '_ew_canvas', None) is None:
            return
        w = max(1, self._ew_canvas.width())
        g = self._ew_align_gutter_px()
        # Flush to canvas right (~2 px) so longitude axis lines up with mosaic below.
        right = 1.0 - 2.0 / w
        if g <= 0:
            self._ew_fig.subplots_adjust(
                left=0.09, right=right, top=0.93, bottom=0.18)
        else:
            self._ew_fig.subplots_adjust(
                left=g / w, right=right, top=0.93, bottom=0.18)
        self._ew_canvas.draw_idle()

    def _sync_ew_xlim_from_mosaic(self) -> None:
        """EW Profile x-axis shows the same corot longitude range as the mosaic FOV."""
        if (getattr(self, '_cor_wrap', None) is None
                or not self._cor_wrap.isVisible()
                or self._mosaic_data is None):
            return
        md = self._mosaic_data
        iw = self._image_widget
        hv = iw.horizontalScrollBar().value()
        vw = iw.longitude_fov_span_px()
        xz, _ = iw.get_zoom()
        if vw <= 0 or xz <= 0:
            return
        px0 = hv / xz
        px1 = (hv + vw) / xz
        hi = md.longitude_extent_hi_deg
        if hi is None:
            hi = float(md.n_long * md.long_interval)
        c0 = float(np.clip(px0 * md.long_interval, 0.0, hi))
        c1 = float(np.clip(px1 * md.long_interval, 0.0, hi))
        self._ew_ax.set_xlim(c0, c1)
        self._ew_canvas.draw_idle()

    def _init_radial_axes(self) -> None:
        ax = self._radial_ax
        ax.set_xlabel(
            'Radius offset from local core at corotating longitude —° (km)',
            fontsize=8)
        ax.set_ylabel('I/F', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.margins(y=0)

    def _init_corot_ew_axes(self) -> None:
        ax = self._ew_ax
        ax.set_xlabel('Corotating longitude (°)', fontsize=8)
        self._update_corot_ew_ylabel()
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    def _update_corot_ew_ylabel(self) -> None:
        if getattr(self, '_chk_corot_use_ewmu', None) is None:
            return
        self._ew_ax.set_ylabel(
            'EW×μ (km)' if self._chk_corot_use_ewmu.isChecked() else 'EW (km)',
            fontsize=8)

    def _sync_axis_tick_options(self) -> None:
        if self._mosaic_data is None:
            return
        self._image_widget.set_axis_tick_options(
            self._chk_lon_ticks.isChecked(),
            self._chk_rad_ticks.isChecked(),
            self._mosaic_data.mean_core_radius,
        )

    def _on_rad_profile_toggled(self, checked: bool) -> None:
        self._rad_wrap.setVisible(checked)
        if checked and self._mosaic_data is not None:
            self._safe_radial_canvas_draw()
        self._balance_plot_splitter()

    def _on_corot_ew_panel_toggled(self, checked: bool) -> None:
        self._cor_wrap.setVisible(checked)
        if checked:
            self._update_corot_ew_ylabel()
            self._ew_canvas.draw_idle()
        self._balance_plot_splitter()
        self._sync_ew_mosaic_layout()
        if self._mosaic_data is not None:
            self._fit_zoom_to_window()
        self._sync_ew_xlim_from_mosaic()

    def _balance_plot_splitter(self) -> None:
        """Resize splitter so visible plot panels share space; mosaic fills the rest."""
        sp = self._plot_splitter
        h = max(250, sp.height())
        rad_on = self._rad_wrap.isVisible()
        cor_on = self._cor_wrap.isVisible()
        if rad_on and cor_on:
            sp.setSizes([int(h * 0.24), int(h * 0.28), int(h * 0.48)])
        elif rad_on:
            sp.setSizes([int(h * 0.32), 0, int(h * 0.68)])
        elif cor_on:
            sp.setSizes([0, int(h * 0.35), int(h * 0.65)])
        else:
            sp.setSizes([0, 0, h])

    def _on_corot_ew_mode_changed(self) -> None:
        self._update_corot_ew_ylabel()
        if self._mosaic_data is not None:
            self._replot_corot_ew_panel()

    def _on_clear_ew_profile(self) -> None:
        """Remove all radial-band overlays; next Ctrl+click starts a new band."""
        self._ew_radial_ranges.clear()
        self._ew_phase = 0
        if self._mosaic_data is not None:
            self._replot_corot_ew_panel()
        else:
            self._reset_ew_plot()
        self.statusBar().showMessage(self._status_bar_hint())

    def _reset_ew_plot(self) -> None:
        """Clear corotating EW axes (new mosaic)."""
        self._ew_ax.cla()
        self._init_corot_ew_axes()
        self._ew_canvas.draw_idle()

    def _column_band_ew(self, md: MosaicData, arr_min: int, arr_max: int
                        ) -> ma.MaskedArray:
        return ma.sum(md.image_ma[arr_min:arr_max + 1, :], axis=0
                    ) * md.radial_interval

    def _column_band_ewmu(
        self, md: MosaicData, arr_min: int, arr_max: int
    ) -> ma.MaskedArray:
        ew = self._column_band_ew(md, arr_min, arr_max)
        em = md.meta['rings_emission_angle']
        mu = np.abs(np.cos(np.radians(em.filled(0.0))))
        return ew * mu

    def _replot_corot_ew_panel(self) -> None:
        """Redraw full corotating EW plot and all remembered radial bands."""
        md = self._mosaic_data
        if md is None:
            return
        self._ew_ax.cla()
        self._init_corot_ew_axes()
        use_mu = self._chk_corot_use_ewmu.isChecked()
        longs = np.arange(md.n_long) * md.long_interval
        if use_mu:
            yfull = md.ew_mu.filled(np.nan)
            stat = f'{md.ewmu_mean:.4f} ± {md.ewmu_std:.4f} km'
            tag = 'Full (EW×μ)'
        else:
            yfull = md.ew.filled(np.nan)
            stat = f'{md.ew_mean:.4f} ± {md.ew_std:.4f} km'
            tag = 'Full (EW)'
        self._ew_ax.plot(
            longs, yfull, color='steelblue', lw=0.8,
            label=f'{tag}  {stat}')
        for i, (arr_min, arr_max) in enumerate(self._ew_radial_ranges):
            c = _EW_BAND_COLOR_CYCLE[i % len(_EW_BAND_COLOR_CYCLE)]
            self._draw_corot_band_curve(md, arr_min, arr_max, use_mu, c)
        self._ew_ax.legend(fontsize=7, loc='upper right')
        self._sync_ew_xlim_from_mosaic()

    def _draw_corot_band_curve(
        self,
        md: MosaicData,
        arr_min: int,
        arr_max: int,
        use_mu: bool,
        color: str,
    ) -> None:
        if use_mu:
            ew_data = self._column_band_ewmu(md, arr_min, arr_max)
        else:
            ew_data = self._column_band_ew(md, arr_min, arr_max)
        longs = np.arange(md.n_long) * md.long_interval
        rel_min = (arr_min - (md.n_radii - 1) / 2.0) * md.radial_interval
        rel_max = (arr_max - (md.n_radii - 1) / 2.0) * md.radial_interval
        valid = ew_data.compressed()
        ew_mean = float(np.mean(valid)) if valid.size > 0 else 0.0
        ew_std = float(np.std(valid)) if valid.size > 0 else 0.0
        self._ew_ax.plot(
            longs, ew_data.filled(np.nan), color=color, lw=0.8,
            label=f'{rel_min:.0f} to {rel_max:.0f} km  '
                  f'{ew_mean:.4f} ± {ew_std:.4f}')

    @staticmethod
    def _mosaic_radial_abs_km_bounds(md: MosaicData) -> tuple[float, float]:
        """Full mosaic radial extent in km (mean core + nominal grid)."""
        rows = np.arange(md.n_radii, dtype=np.float64)
        rel = (rows - (md.n_radii - 1) / 2.0) * md.radial_interval
        abs_r = rel + md.mean_core_radius
        return float(np.min(abs_r)), float(np.max(abs_r))

    @staticmethod
    def _corot_longitude_for_column(md: MosaicData, ix: int) -> float:
        """Corotating longitude (°) at array column ``ix`` (same cap as EW xlim)."""
        hi = md.longitude_extent_hi_deg
        if hi is None:
            hi = float(md.n_long * md.long_interval)
        return float(np.clip(ix * md.long_interval, 0.0, hi))

    def _update_radial_profile_plot(self, ix: int) -> None:
        """Plot I/F vs radius for column ix; axis limits fixed for the mosaic."""
        md = self._mosaic_data
        if md is None or not self._chk_rad_profile.isChecked():
            return
        c = getattr(self, '_radial_canvas', None)
        if c is not None:
            try:
                from PyQt6 import sip
                if sip.isdeleted(c):
                    return
            except Exception:
                pass
        col = md.image_ma[:, ix]
        arr_rows = np.arange(md.n_radii, dtype=np.float64)
        rel = (arr_rows - (md.n_radii - 1) / 2.0) * md.radial_interval
        y = np.asarray(col, dtype=np.float64)
        if hasattr(col, 'mask'):
            m = ma.getmaskarray(col)
            y = np.where(m, np.nan, y)

        abs_lo, abs_hi = MosaicWindow._mosaic_radial_abs_km_bounds(md)
        rel_lo = abs_lo - md.mean_core_radius
        rel_hi = abs_hi - md.mean_core_radius
        r_span = max(abs(rel_lo), abs(rel_hi), 1e-6)
        y_lo, y_hi = md.image_vmin, md.image_vmax
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-6

        ax = self._radial_ax
        line = self._radial_profile_line
        need_new = (
            line is None
            or getattr(line, 'axes', None) is None
            or line.axes is not ax)
        if need_new:
            ax.clear()
            self._init_radial_axes()
            self._radial_profile_line, = ax.plot(rel, y, 'b-', lw=0.9)
        else:
            line.set_data(rel, y)
        ax.set_xlim(-r_span, r_span)
        ax.margins(y=0)
        ax.set_ylim(y_lo, y_hi)
        ax.set_autoscaley_on(False)
        corot_lon = MosaicWindow._corot_longitude_for_column(md, ix)
        ax.set_xlabel(
            f'Radius offset from local core at corotating longitude '
            f'{corot_lon:.2f}° (km)',
            fontsize=8)
        self._safe_radial_canvas_draw()

    def _add_ew_range_to_plot(self, py1: float, py2: float,
                               md: MosaicData) -> None:
        """Add a radial band to the corotating EW plot and remember it."""
        arr1 = self._image_widget.pixel_y_to_arr_row(py1)
        arr2 = self._image_widget.pixel_y_to_arr_row(py2)
        arr_min = max(0, min(arr1, arr2))
        arr_max = min(md.n_radii - 1, max(arr1, arr2))
        self._ew_radial_ranges.append((arr_min, arr_max))
        self._replot_corot_ew_panel()
        if not self._chk_corot_ew.isChecked():
            self._chk_corot_ew.setChecked(True)

    def _build_control_panel(self) -> QWidget:
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(4, 2, 4, 2)
        ctrl_layout.setSpacing(2)

        # ---- Upper: Stretch + Zoom ----
        upper = QWidget()
        upper_h = QHBoxLayout(upper)
        upper_h.setContentsMargins(0, 0, 0, 0)
        upper_h.setSpacing(6)

        # Stretch group: sliders + presets (Reset / Full / Bright)
        stretch_box = QGroupBox('Stretch')
        stretch_outer = QHBoxLayout(stretch_box)
        stretch_outer.setContentsMargins(4, 4, 4, 4)
        stretch_outer.setSpacing(8)
        stretch_form = QFormLayout()
        stretch_form.setHorizontalSpacing(4)

        def _make_stretch_row():
            le = QLineEdit()
            le.setMaximumWidth(70)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 1000)
            row = QWidget()
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(le)
            rh.addWidget(sl, stretch=1)
            return le, sl, row

        self._black_le, self._black_sl, black_row = _make_stretch_row()
        self._white_le, self._white_sl, white_row = _make_stretch_row()
        self._gamma_le, self._gamma_sl, gamma_row = _make_stretch_row()
        stretch_form.addRow('Black:', black_row)
        stretch_form.addRow('White:', white_row)
        stretch_form.addRow('Gamma:', gamma_row)

        self._black_sync = _SyncedSlider(
            self._black_le, self._black_sl, 0.0, 1.0, '%.6f',
            on_change=lambda _: self._apply_stretch())
        self._white_sync = _SyncedSlider(
            self._white_le, self._white_sl, 0.0, 1.0, '%.6f',
            on_change=lambda _: self._apply_stretch())
        self._gamma_sync = _SyncedSlider(
            self._gamma_le, self._gamma_sl, 0.01, 5.0, '%.3f',
            on_change=lambda _: self._apply_stretch())
        self._gamma_sync.set_value(0.5)

        stretch_btn_col = QVBoxLayout()
        stretch_btn_col.setSpacing(4)
        btn_stretch_reset = QPushButton('Reset')
        btn_stretch_full = QPushButton('Full')
        btn_stretch_bright = QPushButton('Bright')
        btn_stretch_reset.setToolTip(
            'Restore default stretch for this image (min black, ~99.5% white).')
        btn_stretch_full.setToolTip(
            'Set black and white to the min and max of valid image pixels.')
        btn_stretch_bright.setToolTip(
            'Same as default auto-stretch but clip the brightest 2% (~98% white).')
        btn_stretch_reset.clicked.connect(self._on_stretch_preset_reset)
        btn_stretch_full.clicked.connect(self._on_stretch_preset_full)
        btn_stretch_bright.clicked.connect(self._on_stretch_preset_bright)
        for b in (btn_stretch_reset, btn_stretch_full, btn_stretch_bright):
            b.setMaximumWidth(72)
            stretch_btn_col.addWidget(b)
        stretch_btn_col.addStretch()

        stretch_outer.addLayout(stretch_form, stretch=1)
        stretch_outer.addLayout(stretch_btn_col)
        upper_h.addWidget(stretch_box, stretch=2)

        # Zoom group
        zoom_box = QGroupBox('Zoom')
        zoom_layout = QVBoxLayout(zoom_box)
        zoom_layout.setSpacing(2)

        def _make_zoom_row(label: str):
            le = QLineEdit()
            le.setMaximumWidth(55)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(1, 1000)
            row = QWidget()
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(QLabel(label))
            rh.addWidget(le)
            rh.addWidget(sl, stretch=1)
            return le, sl, row

        self._xzoom_le, self._xzoom_sl, xz_row = _make_zoom_row('X:')
        self._yzoom_le, self._yzoom_sl, yz_row = _make_zoom_row('Y:')
        zoom_layout.addWidget(xz_row)
        zoom_layout.addWidget(yz_row)

        self._xzoom_sync = self._make_zoom_sync(self._xzoom_le, self._xzoom_sl, axis='x')
        self._yzoom_sync = self._make_zoom_sync(self._yzoom_le, self._yzoom_sl, axis='y')

        btn_row = QHBoxLayout()
        self._zoom_info_lbl = QLabel('1.00x / 1.00x')
        btn_zi = QPushButton('+')
        btn_zo = QPushButton('−')
        btn_zr = QPushButton('Reset')
        btn_sf = QPushButton('Save FOV')
        btn_zi.setMaximumWidth(28)
        btn_zo.setMaximumWidth(28)
        btn_zi.clicked.connect(self._on_zoom_in)
        btn_zo.clicked.connect(self._on_zoom_out)
        btn_zr.clicked.connect(self._on_zoom_reset)
        btn_sf.clicked.connect(self._on_save_fov)
        btn_row.addWidget(self._zoom_info_lbl)
        btn_row.addWidget(btn_zi)
        btn_row.addWidget(btn_zo)
        btn_row.addWidget(btn_zr)
        btn_row.addStretch()
        btn_row.addWidget(btn_sf)
        zoom_layout.addLayout(btn_row)
        upper_h.addWidget(zoom_box, stretch=1)
        ctrl_layout.addWidget(upper)

        # ---- Lower: Info + Color-By ----
        lower = QWidget()
        lower_h = QHBoxLayout(lower)
        lower_h.setContentsMargins(0, 0, 0, 0)
        lower_h.setSpacing(6)

        # Info group — four columns: position | illumination & sampling |
        # orbit & moons | source & EW
        info_box = QGroupBox('Cursor Info')
        info_grid_widget = QWidget()
        info_grid = QGridLayout(info_grid_widget)
        info_grid.setHorizontalSpacing(10)
        info_grid.setVerticalSpacing(0)
        info_grid.setContentsMargins(2, 0, 2, 0)
        info_box_layout = QVBoxLayout(info_box)
        info_box_layout.setContentsMargins(4, 1, 4, 1)
        info_box_layout.setSpacing(0)
        info_box_layout.addWidget(info_grid_widget)

        info_columns: list[list[tuple[str, str]]] = [
            [
                ('corot', 'Corotating longitude:'),
                ('rel_r', 'Relative radius:'),
                ('inert', 'Inertial longitude:'),
                ('core_r', 'Core radius:'),
            ],
            [
                ('incidence', 'Incidence angle:'),
                ('phase', 'Phase angle:'),
                ('emission', 'Emission angle:'),
                ('rad_res', 'Radial resolution:'),
                ('long_res', 'Longitudinal resolution:'),
            ],
            [
                ('true_anomaly', 'True anomaly:'),
                ('long_asc_node', 'Longitude ascending node:'),
                ('long_pericenter', 'Longitude of pericenter:'),
                ('prom_corot', 'Prometheus corotating longitude:'),
                ('prom_rad', 'Prometheus radius − core:'),
                ('pand_corot', 'Pandora corotating longitude:'),
                ('pand_rad', 'Pandora radius − core:'),
            ],
            [
                ('image', 'Source image:'),
                ('date', 'Observation date (UTC):'),
                ('long_ew', 'EW at longitude:'),
                ('long_ewmu', 'EW×μ at longitude:'),
                ('full_ew', 'Full mosaic EW:'),
                ('full_ewmu', 'Full mosaic EW×μ:'),
            ],
        ]
        self._info: dict[str, QLabel] = {}
        for col_idx, col in enumerate(info_columns):
            base = col_idx * 2
            for row_idx, (key, name) in enumerate(col):
                nl = QLabel(name)
                nl.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                vl = QLabel('---')
                mw = 155 if col_idx in (2, 3) else 118
                vl.setMinimumWidth(mw)
                info_grid.addWidget(nl, row_idx, base)
                info_grid.addWidget(vl, row_idx, base + 1)
                self._info[key] = vl
        lower_h.addWidget(info_box, stretch=1)

        # Color-by: 2 columns — none|image; rad|lon res; phase/emission/inertial/true pairs
        colorby_box = QGroupBox('Color By')
        cb_grid = QGridLayout(colorby_box)
        cb_grid.setContentsMargins(4, 2, 4, 2)
        cb_grid.setHorizontalSpacing(12)
        cb_grid.setVerticalSpacing(2)
        self._colorby_group = QButtonGroup()
        colorby_rows: list[list[tuple[str, str]]] = []
        if self._colorby_include_image_number:
            colorby_rows.append([('none', 'None'), ('image_no', 'Image number')])
        else:
            colorby_rows.append([('none', 'None')])
        colorby_rows.extend([
            [('rel_rad_res', 'Radial resolution (rel)'),
             ('rel_ang_res', 'Longitudinal resolution (rel)')],
            [('abs_phase', 'Phase (abs)'), ('rel_phase', 'Phase (rel)')],
            [('abs_emission', 'Emission (abs)'), ('rel_emission', 'Emission (rel)')],
            [('abs_inertial', 'Inertial long (abs)'), ('rel_inertial', 'Inertial long (rel)')],
            [('abs_true_anomaly', 'True anomaly (abs)'), ('rel_true_anomaly', 'True anomaly (rel)')],
        ])
        for row_idx, row in enumerate(colorby_rows):
            if len(row) == 1:
                key, label = row[0]
                btn = QRadioButton(label)
                btn.setProperty('colorby_key', key)
                self._colorby_group.addButton(btn)
                cb_grid.addWidget(btn, row_idx, 0, 1, 2)
                if key == 'none':
                    btn.setChecked(True)
            else:
                for col_idx, (key, label) in enumerate(row):
                    btn = QRadioButton(label)
                    btn.setProperty('colorby_key', key)
                    self._colorby_group.addButton(btn)
                    cb_grid.addWidget(btn, row_idx, col_idx)
                    if key == 'none':
                        btn.setChecked(True)
        self._colorby_group.buttonClicked.connect(self._on_colorby_changed)
        lower_h.addWidget(colorby_box)

        ctrl_layout.addWidget(lower)
        return ctrl

    def _make_zoom_sync(
        self, le: QLineEdit, sl: QSlider, axis: str
    ) -> _SyncedSlider:
        def _on_change(zoom_val: float):
            iw = self._image_widget
            xz, yz = iw.get_zoom()
            if axis == 'x':
                iw.set_zoom(zoom_val, yz)
            else:
                iw.set_zoom(xz, zoom_val)

        class _ZoomSync(_SyncedSlider):
            def _to_slider(self, val):
                return zoom_to_slider(val)
            def _from_slider(self, pos):
                return slider_to_zoom(pos)

        sync = _ZoomSync(le, sl, 0.05, 100.0, '%.2f', on_change=_on_change)
        sync.set_value(1.0)
        return sync

    # ------------------------------------------------------------------ #
    #  Catalog / filtering                                                 #
    # ------------------------------------------------------------------ #

    def _update_filtered_list(
        self, prefer_record: Optional[MosaicRecord] = None
    ) -> None:
        if self._catalog is None:
            return
        self._filtered = self._catalog.filter(self._criteria)
        mosaic_list = getattr(self, '_mosaic_list', None)
        if mosaic_list is None:
            return
        mosaic_list.blockSignals(True)
        mosaic_list.clear()
        for rec in self._filtered:
            item = QListWidgetItem(rec.name)
            item.setData(Qt.ItemDataRole.UserRole, rec)
            mosaic_list.addItem(item)
        mosaic_list.blockSignals(False)

        # Choose which record to show
        target = prefer_record
        if target is None and self._mosaic_data is not None:
            # Try to keep current mosaic
            for rec in self._filtered:
                if rec.name == self._mosaic_data.bundle_host_dir:
                    target = rec
                    break
        if target is None and self._filtered:
            target = self._filtered[0]

        if target is not None:
            for i, rec in enumerate(self._filtered):
                if rec.name == target.name:
                    self._current_idx = i
                    break
            else:
                self._current_idx = 0
            self._load_mosaic_record(self._filtered[self._current_idx])
        else:
            self._current_idx = 0

        self._refresh_list_selection()
        self._update_nav_buttons()

    def _refresh_list_selection(self) -> None:
        if self._catalog is None:
            return
        mosaic_list = getattr(self, '_mosaic_list', None)
        if mosaic_list is None:
            return
        mosaic_list.blockSignals(True)
        for i in range(mosaic_list.count()):
            item = mosaic_list.item(i)
            rec: MosaicRecord = item.data(Qt.ItemDataRole.UserRole)
            is_current = (self._mosaic_data is not None
                          and rec.name == self._mosaic_data.bundle_host_dir)
            font = item.font()
            font.setBold(is_current)
            item.setFont(font)
        if 0 <= self._current_idx < mosaic_list.count():
            mosaic_list.setCurrentRow(self._current_idx)
            mosaic_list.scrollToItem(mosaic_list.currentItem())
        mosaic_list.blockSignals(False)

    def _update_nav_buttons(self) -> None:
        if self._catalog is None:
            return
        btn_prev = getattr(self, '_btn_prev', None)
        btn_next = getattr(self, '_btn_next', None)
        if btn_prev is None or btn_next is None:
            return
        n = len(self._filtered)
        btn_prev.setEnabled(self._current_idx > 0)
        btn_next.setEnabled(self._current_idx < n - 1)

    # ------------------------------------------------------------------ #
    #  Mosaic loading                                                      #
    # ------------------------------------------------------------------ #

    def _set_loading_ui(self, enabled: bool) -> None:
        # Reproj-only windows omit the catalog sidebar; those widgets are absent.
        mosaic_list = getattr(self, '_mosaic_list', None)
        if mosaic_list is not None:
            mosaic_list.setEnabled(enabled)
        btn_filter = getattr(self, '_btn_filter', None)
        if btn_filter is not None:
            btn_filter.setEnabled(enabled)
        btn_new_win = getattr(self, '_btn_new_win', None)
        if btn_new_win is not None:
            btn_new_win.setEnabled(enabled)
        btn_prev = getattr(self, '_btn_prev', None)
        btn_next = getattr(self, '_btn_next', None)
        if enabled:
            self._update_nav_buttons()
        else:
            if btn_prev is not None:
                btn_prev.setEnabled(False)
            if btn_next is not None:
                btn_next.setEnabled(False)

    def _load_mosaic_record(self, record: MosaicRecord) -> None:
        if self._is_loading:
            return
        self._is_loading = True
        self._set_loading_ui(False)
        self.statusBar().showMessage(f'Loading {record.name} …')
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            try:
                label_path = self._catalog.label_path(record)
                md = load_mosaic(label_path)
            except Exception as exc:
                self.statusBar().showMessage(f'Error loading {record.name}: {exc}')
                return
            mode = 'bkg-sub' if self._catalog.bkg_sub else 'no-bkg-sub'
            n = len(self._filtered)
            idx = self._current_idx + 1
            ns = record.notes.strip()
            head = f'{md.obsid} [{ns}]' if ns else md.obsid
            title = f'{head} ({mode}) [{idx} of {n}]'
            self._apply_loaded_mosaic(md, title=title, reproj_initial_zoom=False)
            self._refresh_list_selection()
            self._update_nav_buttons()
        finally:
            QApplication.restoreOverrideCursor()
            self._is_loading = False
            self._set_loading_ui(True)

    def _load_reproj_label(self, label_path: str) -> None:
        if self._is_loading:
            return
        self._is_loading = True
        self._set_loading_ui(False)
        self.statusBar().showMessage('Loading reproj …')
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            try:
                md = load_reproj(label_path)
            except Exception as exc:
                self.statusBar().showMessage(f'Error loading reproj: {exc}')
                QMessageBox.critical(
                    self, 'Reprojected image',
                    f'Could not load reprojected image:\n{exc}')
                return
            title = f'Reproj: {md.obsid} / {md.reproj_name}'
            self._apply_loaded_mosaic(md, title=title, reproj_initial_zoom=True)
        finally:
            QApplication.restoreOverrideCursor()
            self._is_loading = False
            self._set_loading_ui(True)

    def _apply_loaded_mosaic(
        self,
        md: MosaicData,
        *,
        title: str,
        reproj_initial_zoom: bool,
    ) -> None:
        self._mosaic_data = md
        self._clear_info()

        black = (self._default_black if self._default_black is not None
                 else md.black)
        white = (self._default_white if self._default_white is not None
                 else md.white)
        gamma = self._default_gamma

        valid_count = md.image_ma.count()
        if valid_count > 0:
            span_lo = min(float(md.image_vmin), 0.0)
            span_hi = float(md.image_vmax)
        else:
            span_lo, span_hi = 0.0, 1.0
        self._black_sync.set_range(span_lo, span_hi)
        self._white_sync.set_range(span_lo, span_hi)
        self._black_sync.set_value(black)
        self._white_sync.set_value(white)
        self._gamma_sync.set_value(gamma)

        self._image_widget.set_image(
            md.image_ma,
            md.long_interval,
            md.radial_interval,
            longitude_extent_hi_deg=md.longitude_extent_hi_deg,
        )
        self._image_widget.set_stretch(black, white, gamma)

        if md.is_reproj:
            # Full-ring grid is mostly masked at low column indices; default
            # scroll (0) shows empty black until zoom-fit runs. Scroll to valid
            # data immediately so the first paint is meaningful even if the
            # viewport size is still 0 and zoom-fit defers to resize/show.
            self._ensure_reproj_scroll_shows_data(md)

        pixel_ys = show_radii_to_pixel_ys(
            self._show_radii_rel_km,
            md.n_radii, md.radial_interval)
        self._image_widget.set_show_radii(pixel_ys)

        self._on_colorby_changed(self._colorby_group.checkedButton())

        self._info['full_ew'].setText(
            f'{md.ew_mean:.5f} ± {md.ew_std:.5f}')
        self._info['full_ewmu'].setText(
            f'{md.ewmu_mean:.5f} ± {md.ewmu_std:.5f}')

        self._ew_phase = 0
        self._ew_radial_ranges.clear()
        self._rad_wrap.setVisible(self._chk_rad_profile.isChecked())
        self._cor_wrap.setVisible(self._chk_corot_ew.isChecked())
        self._balance_plot_splitter()
        self._sync_ew_mosaic_layout()
        if reproj_initial_zoom:
            self._reproj_open_fit_pending = True
            self._fit_zoom_to_reproj_data()
            # Second pass after the event loop lays out the viewport (first
            # call often sees width/height 0 during __init__).
            QTimer.singleShot(0, self._deferred_reproj_zoom_fit)
            QTimer.singleShot(100, self._deferred_reproj_zoom_fit)
        else:
            self._fit_zoom_to_window()
        self._sync_axis_tick_options()
        self._reset_ew_plot()
        self._replot_corot_ew_panel()
        self._radial_profile_line = None
        self._radial_ax.clear()
        self._init_radial_axes()
        self._safe_radial_canvas_draw()

        self.setWindowTitle(title)
        self.statusBar().showMessage(self._status_bar_hint())

    # ------------------------------------------------------------------ #
    #  Stretch                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _reproj_valid_col_span(
        image_ma: ma.MaskedArray,
    ) -> Optional[tuple[float, float]]:
        """Return (center_col, span_cols) of the valid-longitude span.

        Wrap-aware: when valid data touches both edges of the 360° grid with a
        masked gap between (the reprojection wraps corot 0°), the data span is
        the complement of the largest masked run. The pixmap is a fixed 0-360
        grid with no wrap scrolling, so wrapped data is split on screen: the
        returned center is that of the larger on-screen segment while
        span_cols is the full across-wrap data width.
        """
        col_masked = np.all(ma.getmaskarray(image_ma), axis=0)
        valid = np.where(~col_masked)[0]
        if valid.size == 0:
            return None
        n_long = col_masked.size
        if valid[0] == 0 and valid[-1] == n_long - 1 and col_masked.any():
            masked = np.where(col_masked)[0]
            breaks = np.where(np.diff(masked) > 1)[0]
            starts = np.concatenate(([0], breaks + 1))
            ends = np.concatenate((breaks, [masked.size - 1]))
            k = int(np.argmax(ends - starts))
            gap_lo = int(masked[starts[k]])
            gap_hi = int(masked[ends[k]])
            span = float(n_long - (gap_hi - gap_lo + 1))
            # On-screen segments: [gap_hi+1, n_long-1] and [0, gap_lo-1]
            # (both non-empty since columns 0 and n_long-1 are valid).
            if n_long - 1 - gap_hi >= gap_lo:
                center = (float(gap_hi + 1) + float(n_long - 1)) / 2.0
            else:
                center = float(gap_lo - 1) / 2.0
            return center, span
        return ((float(valid[0]) + float(valid[-1])) / 2.0,
                float(valid[-1] - valid[0] + 1))

    def _ensure_reproj_scroll_shows_data(self, md: MosaicData) -> None:
        """Scroll so the viewport includes at least one unmasked longitude column."""
        span = self._reproj_valid_col_span(md.image_ma)
        if span is None:
            return
        cx, _ = span
        cy = (md.n_radii - 1) / 2.0
        self._image_widget.scroll_to_pixel(cx, cy)

    def _fit_zoom_to_reproj_data(self) -> None:
        """Zoom and scroll to the unmasked longitude span (reproj on 360° grid)."""
        if self._mosaic_data is None:
            return
        md = self._mosaic_data
        iw = self._image_widget
        vw0, vh0 = iw.viewport().width(), iw.viewport().height()
        if vw0 <= 0 or vh0 <= 0:
            self._pending_reproj_fit = True
            return
        self._pending_reproj_fit = False
        span = self._reproj_valid_col_span(md.image_ma)
        if span is None:
            self._fit_zoom_to_window()
            self._sync_zoom_ui()
            return
        center_col, col_range = span
        vw = max(vw0, 400)
        vh = max(vh0, 300)
        x_zoom = float(np.clip(vw / max(col_range * 1.1, 1.0), 0.05, 100.0))
        y_zoom = float(np.clip(vh / max(md.n_radii * 1.05, 1.0), 0.05, 100.0))
        center_row = md.n_radii / 2.0
        iw.set_zoom(x_zoom, y_zoom)
        iw.scroll_to_pixel(center_col, center_row)
        self._sync_zoom_ui()

    def _fit_zoom_to_window(self) -> None:
        """Set X/Y zoom so the entire mosaic fits in the current viewport.

        If the viewport is not yet sized (window not shown), sets
        ``_pending_fit`` so that the next resizeEvent completes the fit.
        """
        if self._mosaic_data is None:
            return
        md = self._mosaic_data
        vw = self._image_widget.viewport().width()
        vh = self._image_widget.viewport().height()
        if vw <= 0 or vh <= 0:
            self._pending_fit = True
            return
        self._pending_fit = False
        x_zoom = min(float(vw) / md.n_long, 100.0)
        y_zoom = min(float(vh) / md.n_radii, 100.0)
        self._image_widget.set_zoom(x_zoom, y_zoom)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, '_pending_fit', False):
            self._fit_zoom_to_window()
        if getattr(self, '_pending_reproj_fit', False):
            self._fit_zoom_to_reproj_data()
        if getattr(self, '_cor_wrap', None) is not None and self._cor_wrap.isVisible():
            self._sync_ew_figure_margins()
            self._sync_ew_xlim_from_mosaic()

    def _apply_stretch(self) -> None:
        b = self._black_sync.get_value()
        w = self._white_sync.get_value()
        g = self._gamma_sync.get_value()
        self._image_widget.set_stretch(b, w, g)

    def _on_stretch_preset_reset(self) -> None:
        """Auto defaults for this mosaic (``md.black`` / ``md.white``, 99.5% white)."""
        md = self._mosaic_data
        if md is None:
            return
        self._black_sync.set_value(md.black)
        self._white_sync.set_value(md.white)
        self._gamma_sync.set_value(self._default_gamma)
        self._apply_stretch()

    def _on_stretch_preset_full(self) -> None:
        """Black / white = min / max over valid pixels."""
        md = self._mosaic_data
        if md is None:
            return
        b, w = md.image_vmin, md.image_vmax
        if w <= b:
            w = b + 1e-6
        self._black_sync.set_value(b)
        self._white_sync.set_value(w)
        self._apply_stretch()

    def _on_stretch_preset_bright(self) -> None:
        """Auto stretch with a lower white point (~98% / clip top 2%)."""
        md = self._mosaic_data
        if md is None:
            return
        b, w = compute_default_stretch(
            md.image_ma,
            white_point_ignore_frac=_STRETCH_BRIGHT_WHITE_IGNORE_FRAC,
        )
        self._black_sync.set_value(b)
        self._white_sync.set_value(w)
        self._apply_stretch()

    # ------------------------------------------------------------------ #
    #  Color-by                                                            #
    # ------------------------------------------------------------------ #

    def _on_colorby_changed(self, btn) -> None:
        if btn is None or self._mosaic_data is None:
            self._image_widget.set_color_column(None)
            return
        key = btn.property('colorby_key')
        md = self._mosaic_data
        col = self._compute_color_column(key, md)
        self._image_widget.set_color_column(col)

    def _compute_color_column(
        self, key: str, md: MosaicData
    ) -> Optional[np.ndarray]:
        if key == 'none':
            return None
        meta = md.meta
        if key == 'image_no':
            if not self._colorby_include_image_number:
                return None
            vals = meta['image_index']
            vals_f = vals.astype(float)
            valid = vals_f.compressed()
            if valid.size == 0:
                return None
            return compute_color_column(
                ma.array(vals_f, mask=ma.getmaskarray(vals)),
                float(valid.min()), float(valid.max()))
        if key in _COLORBY_REL_META_FIELD:
            vals = meta[_COLORBY_REL_META_FIELD[key]]
            valid = vals.compressed()
            if valid.size == 0:
                return None
            return compute_color_column(vals, float(valid.min()), float(valid.max()))
        if key in _COLORBY_ABS_RANGE:
            field, lo, hi = _COLORBY_ABS_RANGE[key]
            vals = meta[field]
            return compute_color_column(vals, lo, hi)
        return None

    # ------------------------------------------------------------------ #
    #  Zoom controls                                                       #
    # ------------------------------------------------------------------ #

    def _on_zoom_in(self) -> None:
        xz, yz = self._image_widget.get_zoom()
        self._image_widget.set_zoom(xz * 1.5, yz * 1.5)
        self._sync_zoom_ui()

    def _on_zoom_out(self) -> None:
        xz, yz = self._image_widget.get_zoom()
        self._image_widget.set_zoom(xz / 1.5, yz / 1.5)
        self._sync_zoom_ui()

    def _on_zoom_reset(self) -> None:
        if (self._mosaic_data is not None
                and self._mosaic_data.is_reproj):
            self._fit_zoom_to_reproj_data()
        else:
            self._fit_zoom_to_window()

    def _on_zoom_changed(self, xz: float, yz: float) -> None:
        """Invoked by TiledImageWidget.zoom_changed signal."""
        self._update_zoom_slider_ranges()
        self._xzoom_sync.set_value(xz)
        self._yzoom_sync.set_value(yz)
        self._zoom_info_lbl.setText(f'{xz:.2f}x / {yz:.2f}x')

    def _update_zoom_slider_ranges(self) -> None:
        if getattr(self, '_image_widget', None) is None:
            return
        min_x, min_y = self._image_widget.get_min_zoom()
        self._xzoom_sync.set_range(min_x, 100.0)
        self._yzoom_sync.set_range(min_y, 100.0)

    def _sync_zoom_ui(self) -> None:
        self._update_zoom_slider_ranges()
        xz, yz = self._image_widget.get_zoom()
        self._xzoom_sync.set_value(xz)
        self._yzoom_sync.set_value(yz)
        self._zoom_info_lbl.setText(f'{xz:.2f}x / {yz:.2f}x')

    # ------------------------------------------------------------------ #
    #  Save FOV                                                            #
    # ------------------------------------------------------------------ #

    def _on_save_fov(self) -> None:
        if self._mosaic_data is None:
            return
        md = self._mosaic_data
        stem = md.reproj_name if md.is_reproj else md.bundle_host_dir
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Field of View',
            f'{stem}_fov.png',
            'PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)',
        )
        if not path:
            return
        qimg = self._image_widget.render_viewport_to_image()
        if not qimg.save(path):
            QMessageBox.warning(self, 'Save FOV', f'Failed to save {path}')

    # ------------------------------------------------------------------ #
    #  Mouse info panel                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_meta_at_ix(
        meta: dict, ix: int, field: str, fmt: str = '%.4f',
    ) -> str:
        arr = meta[field]
        v = arr[ix]
        if ma.is_masked(v):
            return '---'
        return fmt % float(v)

    @staticmethod
    def _fmt_deg(s: str) -> str:
        return f'{s}°' if s != '---' else '---'

    @staticmethod
    def _rel_sat_to_core(radius_s: str, core_str: str) -> str:
        if radius_s == '---' or core_str == '---':
            return '---'
        try:
            return f'{float(radius_s) - float(core_str):.2f} km'
        except ValueError:
            return '---'

    def _on_mouse_moved(self, px: float, py: float, in_bounds: bool) -> None:
        if not in_bounds or self._mosaic_data is None:
            self._clear_info()
            return

        md = self._mosaic_data
        ix = int(np.clip(round(px), 0, md.n_long - 1))
        arr_row = self._image_widget.pixel_y_to_arr_row(py)
        arr_row = int(np.clip(arr_row, 0, md.n_radii - 1))

        corot, rel_r = self._image_widget.pixel_to_physical(px, py)

        # Array value at cursor (fixed-width fields; label uses monospace)
        raw_val = md.image_ma[arr_row, ix]
        if ma.is_masked(raw_val):
            value_str = f'{"masked":>11}'
        else:
            value_str = f'{float(raw_val):11.8f}'

        # Metadata at this column
        meta = md.meta

        inert = self._format_meta_at_ix(meta, ix, 'rings_inertial_ring_longitude', '%.4f')
        incid = self._format_meta_at_ix(meta, ix, 'rings_incidence_angle', '%.3f')
        phase = self._format_meta_at_ix(meta, ix, 'rings_phase_angle', '%.3f')
        emiss = self._format_meta_at_ix(meta, ix, 'rings_emission_angle', '%.3f')
        rad_r = self._format_meta_at_ix(meta, ix, 'rings_radial_resolution', '%.3f')
        lng_r = self._format_meta_at_ix(meta, ix, 'rings_longitudinal_resolution', '%.5f')
        core_s = self._format_meta_at_ix(meta, ix, 'core_radius', '%.2f')
        long_asc = self._format_meta_at_ix(meta, ix, 'longitude_ascending_node', '%.4f')
        long_peri = self._format_meta_at_ix(meta, ix, 'longitude_pericenter', '%.4f')
        true_anom = self._format_meta_at_ix(meta, ix, 'true_anomaly', '%.4f')
        prom_c = self._format_meta_at_ix(meta, ix, 'corotating_longitude_prometheus', '%.4f')
        prom_r = self._format_meta_at_ix(meta, ix, 'radius_prometheus', '%.2f')
        pand_c = self._format_meta_at_ix(meta, ix, 'corotating_longitude_pandora', '%.4f')
        pand_r = self._format_meta_at_ix(meta, ix, 'radius_pandora', '%.2f')

        # Image name
        if md.is_reproj:
            img_name = MosaicWindow._format_reproj_display_name(md.reproj_name)
        else:
            img_idx_arr = meta['image_index']
            if not ma.is_masked(img_idx_arr[ix]):
                img_idx = int(img_idx_arr[ix])
                lidvid = md.image_table[img_idx]
                raw_name = lidvid_to_reproj_name(lidvid)
                img_name = MosaicWindow._format_reproj_display_name(raw_name)
            else:
                img_name = '---'

        # Date
        tdb_arr = meta['rings_observed_event_tdb']
        if not ma.is_masked(tdb_arr[ix]):
            date_str = tdb_to_utc_str(float(tdb_arr[ix]))
        else:
            date_str = '---'

        # EW at this column
        ew_v = md.ew[ix]
        ewmu_v = md.ew_mu[ix]
        ew_str = f'{float(ew_v):.5f}' if not ma.is_masked(ew_v) else '---'
        ewmu_str = f'{float(ewmu_v):.5f}' if not ma.is_masked(ewmu_v) else '---'

        x_str = f'{px:8.2f}'
        y_str = f'{py:7.2f}'
        self._cursor_status_lbl.setText(
            f'X: {x_str}  Y: {y_str}  Value: {value_str}')
        self._info['rel_r'].setText(f'{rel_r:.2f} km')
        self._info['corot'].setText(f'{corot:.4f}°')
        self._info['inert'].setText(self._fmt_deg(inert))
        self._info['incidence'].setText(self._fmt_deg(incid))
        self._info['phase'].setText(self._fmt_deg(phase))
        self._info['emission'].setText(self._fmt_deg(emiss))
        self._info['rad_res'].setText(
            f'{rad_r} km/px' if rad_r != '---' else '---')
        self._info['long_res'].setText(
            f'{lng_r} deg/px' if lng_r != '---' else '---')
        self._info['core_r'].setText(
            f'{core_s} km' if core_s != '---' else '---')
        self._info['true_anomaly'].setText(self._fmt_deg(true_anom))
        self._info['long_asc_node'].setText(self._fmt_deg(long_asc))
        self._info['long_pericenter'].setText(self._fmt_deg(long_peri))
        self._info['prom_corot'].setText(self._fmt_deg(prom_c))
        self._info['prom_rad'].setText(
            self._rel_sat_to_core(prom_r, core_s))
        self._info['pand_corot'].setText(self._fmt_deg(pand_c))
        self._info['pand_rad'].setText(
            self._rel_sat_to_core(pand_r, core_s))
        self._info['image'].setText(img_name)
        self._info['date'].setText(date_str)
        self._info['long_ew'].setText(ew_str)
        self._info['long_ewmu'].setText(ewmu_str)

        if self._chk_rad_profile.isChecked():
            self._update_radial_profile_plot(ix)

    @staticmethod
    def _format_reproj_display_name(name: str) -> str:
        n = name.strip()
        if n.endswith('_reproj_img'):
            return n[: -len('_reproj_img')]
        return n

    def _clear_info(self) -> None:
        self._cursor_status_lbl.setText('')
        for lbl in self._info.values():
            lbl.setText('---')

    # ------------------------------------------------------------------ #
    #  Right-click: open reproj image                                     #
    # ------------------------------------------------------------------ #

    def _on_right_click(self, px: float, py: float) -> None:
        if self._reproj_mode:
            return
        if self._mosaic_data is None:
            return
        md = self._mosaic_data
        ix = int(np.clip(round(px), 0, md.n_long - 1))
        img_idx_arr = md.meta['image_index']
        if ma.is_masked(img_idx_arr[ix]):
            self.statusBar().showMessage('No image at this longitude.')
            return
        img_idx = int(img_idx_arr[ix])
        lidvid = md.image_table[img_idx]
        reproj_name = lidvid_to_reproj_name(lidvid)
        label_path = os.path.join(
            self._bundle_path, 'data_reproj_img',
            md.bundle_host_dir, f'{reproj_name}.lblx')
        if not os.path.isfile(label_path):
            self.statusBar().showMessage(
                f'Reproj image not found: {label_path}')
            return

        win = MosaicWindow(
            catalog=None,
            bundle_path=self._bundle_path,
            reproj_label_path=label_path,
        )
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._retain_top_level_window(win)
        win.show()

    # ------------------------------------------------------------------ #
    #  Ctrl+click: EW selection                                           #
    # ------------------------------------------------------------------ #

    def _on_ctrl_click(self, px: float, py: float) -> None:
        if self._mosaic_data is None:
            return
        md = self._mosaic_data
        _, rel_r = self._image_widget.pixel_to_physical(px, py)

        if self._ew_phase == 0:
            self._ew_phase = 1
            self._ew_first_py = py
            self.statusBar().showMessage(
                f'Ctrl+click to select upper radial boundary '
                f'(lower: rel_r={rel_r:.1f} km). '
                f'ESC to cancel.')
        else:
            self._ew_phase = 0
            self._add_ew_range_to_plot(self._ew_first_py, py, md)
            self.statusBar().showMessage(self._status_bar_hint())

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self._ew_phase != 0:
            self._ew_phase = 0
            self.statusBar().showMessage(self._status_bar_hint())
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    #  Sidebar button handlers                                             #
    # ------------------------------------------------------------------ #

    def _on_filter(self) -> None:
        if self._catalog is None:
            return
        dlg = FilterDialog(self._criteria, self._catalog.bkg_sub, parent=self)
        if dlg.exec() == FilterDialog.DialogCode.Accepted:
            self._criteria = dlg.get_criteria()
            self._update_filtered_list()

    def _on_new_window(self) -> None:
        if self._catalog is None:
            return
        initial_record = None
        if (self._filtered
                and 0 <= self._current_idx < len(self._filtered)):
            initial_record = self._filtered[self._current_idx]
        win = MosaicWindow(
            catalog=self._catalog,
            bundle_path=self._bundle_path,
            show_radii=list(self._show_radii_rel_km),
            criteria=self._criteria.copy(),
            initial_record=initial_record,
            initial_black=self._black_sync.get_value(),
            initial_white=self._white_sync.get_value(),
            initial_gamma=self._gamma_sync.get_value(),
        )
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._retain_top_level_window(win)
        win.show()

    def _on_prev(self) -> None:
        if self._catalog is None:
            return
        if self._current_idx > 0:
            self._current_idx -= 1
            self._load_mosaic_record(self._filtered[self._current_idx])

    def _on_next(self) -> None:
        if self._catalog is None:
            return
        if self._current_idx < len(self._filtered) - 1:
            self._current_idx += 1
            self._load_mosaic_record(self._filtered[self._current_idx])

    def _on_list_item_clicked(self, item: QListWidgetItem) -> None:
        rec: MosaicRecord = item.data(Qt.ItemDataRole.UserRole)
        for i, r in enumerate(self._filtered):
            if r.name == rec.name:
                self._current_idx = i
                break
        self._load_mosaic_record(rec)
