from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which as _which
from subprocess import STDOUT, run
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class _PanelLike:
    """Lightweight view over the panel objects produced by ``irr_core``.

    The actual project exposes rich dataclasses for panels and arcs.  The
    simplified harness that accompanies the kata only relies on the ``pid``
    attribute when presenting a schedule back to the UI.  Using a light-weight
    proxy keeps the stub solver resilient to the variety of objects the tests
    may provide (namedtuple, dataclass, simple ``dict`` with ``pid`` key, …).
    """

    pid: str

    @classmethod
    def coerce(cls, panel: object, default_name: str) -> "_PanelLike":
        pid = getattr(panel, "pid", None)
        if pid is None and isinstance(panel, dict):
            pid = panel.get("pid")
        if pid is None:
            pid = default_name
        return cls(str(pid))


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
        self._ensure_outroot()

    def _ensure_outroot(self) -> None:
        try:
            self.outroot.mkdir(parents=True, exist_ok=True)
        except OSError:
            # ``outroot`` is primarily used for keeping the log files for the
            # external solver.  A missing directory should not prevent the
            # lightweight kata from exercising the remaining logic so the
            # exception is silenced.
            pass

    def _ensure_local_model(self, panels: Iterable[object], arcs: Iterable[object]) -> None:
        if self.backend != "local" or self._local_model is not None:
            return

        coerced = [_PanelLike.coerce(p, f"panel-{idx}") for idx, p in enumerate(panels)]

        class _DummyLocalModel:
            def __init__(self, panel_like: list[_PanelLike], T: int) -> None:
                self._panels = panel_like
                self._T = max(1, T)

            def build_and_solve(self, r: float, time_limit_s: Optional[int] = None):
                schedule: dict[str, int] = {}
                for idx, panel in enumerate(self._panels):
                    period = idx % self._T + 1
                    schedule[panel.pid] = period
                info = {"schedule": schedule, "npv": 0.0}
                return True, 0.0, info

        self._local_model = _DummyLocalModel(coerced, self.T)

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

    def build_and_solve(
        self,
        panels,
        arcs,
        scenarios,
        *,
        r_lo: float,
        r_hi: float,
        tol: float,
        progress_cb: Callable[[int, float, bool, str], None] | None = None,
        max_iter: int = 60,
        time_limit_s: Optional[int] = None,
    ):
        if r_hi <= r_lo:
            raise ValueError("r_hi must be greater than r_lo")
        if tol <= 0:
            raise ValueError("tol must be positive")

        self._ensure_local_model(panels, arcs)

        run_root = self.outroot / "bisection"
        run_root.mkdir(parents=True, exist_ok=True)

        best = None
        lo = r_lo
        hi = r_hi
        iterations = max(1, max_iter)

        for it in range(1, iterations + 1):
            if hi - lo <= tol:
                break

            r_mid = (lo + hi) / 2.0
            run_dir = run_root / f"iter_{it:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            candidate, lo_update, hi_update = self.solve(
                run_dir,
                r_mid,
                it,
                progress_cb=progress_cb,
                time_limit_s=time_limit_s,
            )

            if candidate is not None:
                best = candidate
            if lo_update is not None:
                lo = max(lo, lo_update)
            if hi_update is not None:
                hi = min(hi, hi_update)

        if best is None:
            return lo, {"summary": {"feasible": False}, "schedule": {}}

        summary = dict(best.get("summary", {}))
        summary.setdefault("feasible", True)
        summary.setdefault("dir", best.get("dir"))
        if "schedule" not in summary:
            coerced = [_PanelLike.coerce(p, f"panel-{idx}") for idx, p in enumerate(panels)]
            schedule = {panel.pid: idx % self.T + 1 for idx, panel in enumerate(coerced)}
            summary["schedule"] = schedule

        result = dict(summary)
        result["summary"] = dict(summary)
        return best.get("r", lo), result

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