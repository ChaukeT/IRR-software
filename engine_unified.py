from __future__ import annotations

from pathlib import Path
from shutil import which as _which
from subprocess import STDOUT, run
from typing import Callable, Optional


class IRRBisector:
    """Minimal stub of the bisection based IRR engine.

    The original project exposes a significantly larger surface area.  Only the
    pieces that are required by :mod:`engine_adapter` are implemented here.
    """

    def __init__(
        self,
        *,
        outroot: Path,
        econ,
        T: int,
        ROM_cap: float,
        backend: str = "local",
        solver_cmd: str = "",
        mpiprocs: int = 1,
    ) -> None:
        self.outroot = Path(outroot)
        self.econ = econ
        self.T = T
        self.ROM_cap = ROM_cap
        self.backend = backend
        self.solver_cmd = solver_cmd
        self.mpiprocs = mpiprocs
        self._local_model = None  # populated lazily by callers from irr_local_solver

    def solve(
        self,
        run_dir: Path,
        r_mid: float,
        it: int,
        progress_cb: Callable[[int, float, bool, str], None] | None = None,
        time_limit_s: Optional[int] = None,
    ):
        """Solve a single iteration of the bisection search.

        Returns a tuple ``(best, lo, hi)`` mimicking the behaviour of the real
        engine.  Only the pieces that are relied on by tests are implemented –
        the local solver branch delegates to ``self._local_model`` when
        available while the SMPS branch launches the external solver.
        """

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
        elif self.backend == "local" and self._local_model is not None:
            feasible, npv, info = self._local_model.build_and_solve(  # type: ignore[union-attr]
                r_mid, time_limit_s=time_limit_s
            )
            if progress_cb:
                progress_cb(it, r_mid, feasible, f"NPV={npv:.2f}")
            if feasible:
                best = {"r": r_mid, "dir": str(run_dir), "summary": info}
                lo = r_mid
            else:
                hi = r_mid
        else:
            raise ValueError(f"Unsupported backend '{self.backend}'")

        return best, lo, hi

    def _solve_smps(self, run_dir: Path) -> tuple[bool, Path | None, str | None]:
        log_path = run_dir / "solution.log"
        if not self.solver_cmd:
            return False, None, "No solver command configured"

        cmd = [
            "mpirun",
            "-np",
            str(self.mpiprocs),
            self.solver_cmd,
            "model.cor",
            "model.tim",
            "model.sto",
        ]
        if not _which("mpirun") and _which("mpiexec"):
            cmd[0] = "mpiexec"

        try:
            with log_path.open("w", encoding="utf-8", newline="\n") as logf:
                run(cmd, cwd=run_dir, stdout=logf, stderr=STDOUT, check=False, text=True)
            feasible = self._parse_solution(log_path)
            return feasible, log_path, None
        except Exception as exc:  # pragma: no cover - defensive: subprocess/IO failure
            return False, None, f"{type(exc).__name__}: {exc}"

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


def make_engine(
    outroot: str | Path,
    econ,
    T: int,
    ROM_cap: float,
    *,
    backend: str = "local",
    solver_cmd: str = "",
    mpiprocs: int = 1,
) -> IRRBisector:
    return IRRBisector(
        outroot=Path(outroot),
        econ=econ,
        T=T,
        ROM_cap=ROM_cap,
        mpiprocs=mpiprocs,
        solver_cmd=solver_cmd,
        backend=backend,
    )