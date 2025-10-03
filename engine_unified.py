from __future__ import annotations

class IRRBisector:
	# ... other methods ...

	def solve(self, run_dir, r_mid, it, progress_cb=None, time_limit_s=None):
		best = None
		lo = None
		hi = None

		if self.backend == "smps":
			feasible, log_path, err = self._solve_smps(run_dir)
			if progress_cb:
				progress_cb(it, r_mid, feasible, str(log_path if feasible else err))
			if feasible:
				best = {"r": r_mid, "dir": str(run_dir), "summary": {"feasible": True}}
				lo = r_mid
			else:
				hi = r_mid
		elif self.backend == "local":
			feasible, npv, info = self._local_model.build_and_solve(r_mid, time_limit_s=time_limit_s) # type: ignore
			if progress_cb:
				progress_cb(it, r_mid, feasible, f"NPV={npv:.2f}")
def _solve_smps(self, run_dir: Path) -> tuple[bool, Path | None, str | None]:
	log_path = run_dir / "solution.log"
	cmd = ["mpirun", "-np", str(self.mpiprocs), self.solver_cmd, "model.cor", "model.tim", "model.sto"]
	if not _which("mpirun") and _which("mpiexec"):
		cmd[0] = "mpiexec"
	try:
		with log_path.open("w", encoding="utf-8", newline="\n") as logf:
			import subprocess
			subprocess.run(cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT, check=False, text=True)
		feasible = self._parse_solution(log_path)
		return feasible, log_path, None
	except Exception as e:
		return False, None, f"{type(e).__name__}: {e}"
cmd = ["mpirun", "-np", str(self.mpiprocs), self.solver_cmd, "model.cor", "model.tim", "model.sto"]
def _parse_solution(self, logfile: Path) -> bool:
	try:
		txt = logfile.read_text(encoding="utf-8", errors="ignore").upper()
	except Exception:
		return False
	if any(tok in txt for tok in ["INFEASIBLE", "UNBOUNDED", "FAILED", "ERROR"]):
		return False
	if any(tok in txt for tok in ["OPTIMAL", "CONVERGED", "PRIMAL/DUAL", "SOLUTION FOUND"]):
		return True
	return False

from pathlib import Path

def _parse_solution(self, logfile: Path) -> bool:
	try:
		txt = logfile.read_text(encoding="utf-8", errors="ignore").upper()
	except Exception:
		return False
	if any(tok in txt for tok in ["INFEASIBLE", "UNBOUNDED", "FAILED", "ERROR"]):
		return False
	if any(tok in txt for tok in ["OPTIMAL", "CONVERGED", "PRIMAL/DUAL", "SOLUTION FOUND"]):
		return True
	return False

def _which(prog: str) -> str | None:
	from shutil import which
	return which(prog)

def make_engine(outroot: str | Path, econ: Econ, T: int, ROM_cap: float, *, backend: str = "local", solver_cmd: str = "", mpiprocs: int = 1) -> IRRBisector:
	return IRRBisector(outroot=Path(outroot), econ=econ, T=T, ROM_cap=ROM_cap, mpiprocs=mpiprocs, solver_cmd=solver_cmd, backend=backend)