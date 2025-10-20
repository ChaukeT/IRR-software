from __future__ import annotations
from typing import Dict
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


class ScheduleTable(QTableWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setColumnCount(2)
		self.setHorizontalHeaderLabels(["Panel ID", "Period"])

	def set_schedule(self, sched: Dict[str, int]):
		items = sorted(sched.items(), key=lambda kv: kv[1])
		self.setRowCount(len(items))
		for r, (pid, t) in enumerate(items):
			self.setItem(r, 0, QTableWidgetItem(pid))
			self.setItem(r, 1, QTableWidgetItem(str(t)))
		self.resizeColumnsToContents()