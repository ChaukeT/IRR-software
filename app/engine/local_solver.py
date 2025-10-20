from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
import os
from ortools.sat.python import cp_model

class IRRSolver:
	def __init__(self, panels, arcs, pid_index, V, mc, ROM_cap, P, T):
		self.panels = panels
		self.arcs = arcs
		self.pid_index = pid_index
		self.V = V
		self.mc = mc
		self.ROM_cap = ROM_cap
		self.P = P
		self.T = T

	def build_and_solve(self, r: float, time_limit_s: Optional[int] = 60) -> Tuple[bool, float, Dict[str, Any]]:
		m = cp_model.CpModel()
		P, T = self.P, self.T
		x = [[m.NewBoolVar(f"x_{p}_{t}") for t in range(T)] for p in range(P)]
		for p in range(P):
			m.Add(sum(x[p][t] for t in range(T)) <= 1)
		tonnes = [p.tonnes for p in self.panels]
		for t in range(T):
			m.Add(sum(int(round(tonnes[p])) * x[p][t] for p in range(P)) <= int(round(self.ROM_cap)))
		arcs_idx = [(self.pid_index[a.pred], self.pid_index[a.succ]) for a in self.arcs if a.pred in self.pid_index and a.succ in self.pid_index]
		for (i, j) in arcs_idx:
			for t in range(T):
				m.Add(sum(x[i][tau] for tau in range(t+1)) >= sum(x[j][tau] for tau in range(t+1)))
		df = np.array([(1.0 / ((1.0 + r) ** (t+1))) for t in range(T)], dtype=float)
		coeff = np.zeros((P, T), dtype=float)
		for p in range(P):
			for t in range(T):
				coeff[p, t] = (self.V[p, t] - self.mc[p]) * df[t]
		scale = 100
		int_coeff = [[int(round(scale * coeff[p, t])) for t in range(T)] for p in range(P)]
		obj_terms = [int_coeff[p][t] * x[p][t] for p in range(P) for t in range(T) if int_coeff[p][t] != 0]
		m.Maximize(sum(obj_terms))
		solver = cp_model.CpSolver()
		if time_limit_s is not None:
			solver.parameters.max_time_in_seconds = float(time_limit_s)
			solver.parameters.num_search_workers = max(1, os.cpu_count() or 1)
			solver.parameters.log_search_progress = False
		status = solver.Solve(m)
		best_val = solver.ObjectiveValue() / scale
		feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and best_val >= 0.0
		sched = {}
		if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
			for p in range(P):
				for t in range(T):
					if solver.Value(x[p][t]) == 1:
						sched[self.panels[p].pid] = t+1
						break
		info = {"status": int(status), "npv": best_val, "schedule": sched}
		return feasible, best_val, info

# ---------- convenience ----------

def print_progress(it: int, r: float, feasible: bool, msg: str):
	status = "FEAS" if feasible else "INFEAS"
	print(f"it={it:02d} r={r:0.4f} {status} {msg}")