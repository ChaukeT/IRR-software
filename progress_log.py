from __future__ import annotations
from PySide6.QtWidgets import QTextEdit


class ProgressLog(QTextEdit):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setReadOnly(True)
		self.setLineWrapMode(QTextEdit.NoWrap)

	def append(self, msg: str):
		self.moveCursor(self.textCursor().End)
		super().append(msg)

	def clear(self):
		super().clear()

	def progress_cb(self, it: int, r: float, feasible: bool, msg: str):
		status = "FEAS" if feasible else "INFEAS"
		self.append(f"it={it:02d} r={r:0.4f} {status} {msg}")