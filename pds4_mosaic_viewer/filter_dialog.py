"""Filter dialog for mosaic selection criteria."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QLocale
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QVBoxLayout,
    QWidget,
)

from catalog import FilterCriteria


class FilterDialog(QDialog):
    """Modal dialog for editing mosaic filter criteria."""

    def __init__(
        self,
        criteria: FilterCriteria,
        bkg_sub: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('Filter Mosaics')
        self._bkg_sub = bkg_sub
        self._setup_ui()
        self._populate(criteria)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ---- Quality ----
        quality_box = QGroupBox('Quality Filters')
        qlay = QVBoxLayout(quality_box)

        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel('Nav Quality:'))
        self._nav_g = QCheckBox('Good (G)')
        self._nav_f = QCheckBox('Fair (F)')
        self._nav_p = QCheckBox('Poor (P)')
        for cb in (self._nav_g, self._nav_f, self._nav_p):
            nav_row.addWidget(cb)
        nav_row.addStretch()
        qlay.addLayout(nav_row)

        if self._bkg_sub:
            bkgnd_row = QHBoxLayout()
            bkgnd_row.addWidget(QLabel('Bkgnd Quality:'))
            self._bkgnd_g = QCheckBox('Good (G)')
            self._bkgnd_f = QCheckBox('Fair (F)')
            self._bkgnd_p = QCheckBox('Poor (P)')
            for cb in (self._bkgnd_g, self._bkgnd_f, self._bkgnd_p):
                bkgnd_row.addWidget(cb)
            bkgnd_row.addStretch()
            qlay.addLayout(bkgnd_row)
        else:
            self._bkgnd_g = self._bkgnd_f = self._bkgnd_p = None

        layout.addWidget(quality_box)

        # ---- Numeric filters ----
        numeric_box = QGroupBox('Numeric Filters (leave blank for no constraint)')
        form = QFormLayout(numeric_box)

        self._min_rad_res = QLineEdit()
        self._max_rad_res = QLineEdit()
        self._min_long_res = QLineEdit()
        self._max_long_res = QLineEdit()
        self._min_prom = QLineEdit()
        self._max_prom = QLineEdit()
        self._min_pand = QLineEdit()
        self._max_pand = QLineEdit()

        for le in (
            self._min_rad_res, self._max_rad_res,
            self._min_long_res, self._max_long_res,
            self._min_prom, self._max_prom,
            self._min_pand, self._max_pand,
        ):
            v = QDoubleValidator()
            v.setLocale(QLocale.c())
            v.setNotation(QDoubleValidator.Notation.StandardNotation)
            v.setRange(-1e15, 1e15, 12)
            le.setValidator(v)

        form.addRow('Radial resolution (km):',
                    self._range_row(self._min_rad_res, self._max_rad_res))
        form.addRow('Longitudinal resolution (deg):',
                    self._range_row(self._min_long_res, self._max_long_res))
        form.addRow(
            'Prometheus to core (km):',
            self._range_row(self._min_prom, self._max_prom))
        form.addRow(
            'Pandora to core (km):',
            self._range_row(self._min_pand, self._max_pand))

        layout.addWidget(numeric_box)

        # ---- Buttons ----
        btn_box = QDialogButtonBox()
        self._apply_btn = btn_box.addButton(
            'Apply', QDialogButtonBox.ButtonRole.AcceptRole)
        self._reset_btn = btn_box.addButton(
            'Reset', QDialogButtonBox.ButtonRole.ResetRole)
        btn_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_accept_clicked)
        btn_box.rejected.connect(self.reject)
        self._reset_btn.clicked.connect(self._reset)
        layout.addWidget(btn_box)

    @staticmethod
    def _range_row(lo: QLineEdit, hi: QLineEdit) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel('Min:'))
        lo.setMaximumWidth(80)
        h.addWidget(lo)
        h.addWidget(QLabel('Max:'))
        hi.setMaximumWidth(80)
        h.addWidget(hi)
        h.addStretch()
        return w

    def _populate(self, c: FilterCriteria) -> None:
        self._nav_g.setChecked(c.nav_quality_g)
        self._nav_f.setChecked(c.nav_quality_f)
        self._nav_p.setChecked(c.nav_quality_p)
        if self._bkg_sub and self._bkgnd_g is not None:
            self._bkgnd_g.setChecked(c.bkgnd_quality_g)
            self._bkgnd_f.setChecked(c.bkgnd_quality_f)
            self._bkgnd_p.setChecked(c.bkgnd_quality_p)
        _s = lambda v: '' if v is None else str(v)
        self._min_rad_res.setText(_s(c.min_radial_res))
        self._max_rad_res.setText(_s(c.max_radial_res))
        self._min_long_res.setText(_s(c.min_long_res))
        self._max_long_res.setText(_s(c.max_long_res))
        self._min_prom.setText(_s(c.min_prometheus_dist))
        self._max_prom.setText(_s(c.max_prometheus_dist))
        self._min_pand.setText(_s(c.min_pandora_dist))
        self._max_pand.setText(_s(c.max_pandora_dist))

    def _reset(self) -> None:
        self._populate(FilterCriteria())

    def _on_accept_clicked(self) -> None:
        err = self._numeric_fields_error()
        if err:
            QMessageBox.warning(self, 'Filter', err)
            return
        self.accept()

    def _numeric_fields_error(self) -> str:
        fields = (
            ('Min radial resolution', self._min_rad_res),
            ('Max radial resolution', self._max_rad_res),
            ('Min longitudinal resolution', self._min_long_res),
            ('Max longitudinal resolution', self._max_long_res),
            ('Min Prometheus distance', self._min_prom),
            ('Max Prometheus distance', self._max_prom),
            ('Min Pandora distance', self._min_pand),
            ('Max Pandora distance', self._max_pand),
        )
        for label, w in fields:
            txt = w.text().strip()
            if not txt:
                continue
            try:
                float(txt)
            except ValueError:
                return f'Invalid number in {label}.'
        return ''

    def get_criteria(self) -> FilterCriteria:
        def _p(w: QLineEdit) -> Optional[float]:
            txt = w.text().strip()
            return float(txt) if txt else None

        c = FilterCriteria()
        c.nav_quality_g = self._nav_g.isChecked()
        c.nav_quality_f = self._nav_f.isChecked()
        c.nav_quality_p = self._nav_p.isChecked()
        if self._bkg_sub and self._bkgnd_g is not None:
            c.bkgnd_quality_g = self._bkgnd_g.isChecked()
            c.bkgnd_quality_f = self._bkgnd_f.isChecked()
            c.bkgnd_quality_p = self._bkgnd_p.isChecked()
        c.min_radial_res = _p(self._min_rad_res)
        c.max_radial_res = _p(self._max_rad_res)
        c.min_long_res = _p(self._min_long_res)
        c.max_long_res = _p(self._max_long_res)
        c.min_prometheus_dist = _p(self._min_prom)
        c.max_prometheus_dist = _p(self._max_prom)
        c.min_pandora_dist = _p(self._min_pand)
        c.max_pandora_dist = _p(self._max_pand)
        return c
