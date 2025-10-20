from __future__ import annotations
from pathlib import Path
from PyQt5.QtWidgets import (
        QWidget, QPushButton, QGroupBox, QVBoxLayout, QFormLayout, QLabel, QFileDialog, QMessageBox
)
# Import custom widgets/classes
from .progress_log import ProgressLog
from .schedule_table import ScheduleTable
from ..io import load_blockmodel_csv
# Assume irr_core.engine.Engine is imported elsewhere or passed in

class MainWindow(QWidget):
        def __init__(self, engine, backend, spin_T, rom_cap, r_lo, r_hi, tol, parent=None):
                super().__init__(parent)
                self.engine = engine
                self.backend = backend
                self.spin_T = spin_T
                self.rom_cap = rom_cap
                self.r_lo = r_lo
                self.r_hi = r_hi
                self.tol = tol
                self.panels = None
                self.arcs = None
                self.blockmodel_path = None

                # Left: controls
                left_box = QGroupBox("Controls")
                left_layout = QFormLayout(left_box)
                self.btn_run = QPushButton("Run IRR search")
                self.btn_run.clicked.connect(self.on_run)
                self.btn_run.setEnabled(False)
                left_layout.addRow(self.btn_run)

                # Right: results
                right_box = QGroupBox("Results")
                right_layout = QVBoxLayout(right_box)
                self.lbl_status = QLabel("No run yet")
                self.log = ProgressLog()
                self.table = ScheduleTable()
                right_layout.addWidget(self.lbl_status)
                right_layout.addWidget(self.log, 2)
                right_layout.addWidget(self.table, 3)

                # Main layout
                root = QVBoxLayout(self)
                root.addWidget(left_box, 0)
                root.addWidget(right_box, 1)

        # --------------
        # Slots
        # --------------
        def on_load(self):
                path, _ = QFileDialog.getOpenFileName(self, "Select blockmodel CSV", "", "CSV files (*.csv)")
                if not path:
                        return
                try:
                        panels, idx_map, meta, rock_by = load_blockmodel_csv(Path(path))
                        from irr_core.modelio import build_slope_arcs
                        arcs = build_slope_arcs(idx_map, meta, per_material=True, rock_by=rock_by)
                        self.panels, self.arcs = panels, arcs
                        self.blockmodel_path = Path(path)
                        self.btn_run.setEnabled(True)
                        self.log.append(f"Loaded {len(panels)} panels; built {len(arcs)} arcs from {self.blockmodel_path}")
                except Exception as e:
                        QMessageBox.critical(self, "Load error", str(e))

        def on_run(self):
                if not self.panels or not self.arcs:
                        QMessageBox.warning(self, "No data", "Load a blockmodel first")
                        return
                # build econ placeholder — replace with real values or load from a JSON later
                from irr_core.econ import Econ
                econ = Econ( # TODO: replace with real site values
                        recovery=0.9, au_price_g=1.2, cu_price_t=9000.0, mining_cost_ore=40.0,
                        processing_cost=15.0, waste_costs={"Soil": 15.0, "Weathered": 20.0, "Fresh": 25.0}
                )
                # run
                self.log.clear()
                self.lbl_status.setText("Running…")
                info = self.engine.run(
                        panels=self.panels,
                        arcs=self.arcs,
                        econ=econ,
                        backend=self.backend.currentText(),
                        T=self.spin_T.value(),
                        ROM_cap=float(self.rom_cap.value()),
                        r_lo=float(self.r_lo.value()),
                        r_hi=float(self.r_hi.value()),
                        tol=float(self.tol.value()),
                        progress_cb=self.log.progress_cb,
                )
                if info.get("ok"):
                        r_star = info.get("r", 0.0)
                        self.lbl_status.setText(f"IRR*: {r_star:.4f}")
                        sched = info.get("schedule", {})
                        self.table.set_schedule(sched)
                        self.log.append("Done.")
                else:
                        self.lbl_status.setText("Run failed")
