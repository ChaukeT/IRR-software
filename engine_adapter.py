from __future__ import annotations
from typing import List, Dict, Any, Callable
from irr_core.engine_unified import make_engine
from irr_core.modelio import Panel, Arc
from irr_core.scenarios import Scenario
from irr_core.econ import Econ


class EngineAdapter:
    def run(
        self,
        panels: List[Panel], arcs: List[Arc], econ: Econ,
        backend: str, T: int, ROM_cap: float,
        r_lo: float, r_hi: float, tol: float,
        progress_cb: Callable[[int, float, bool, str], None] | None,
        scenarios: List[Scenario] | None = None,
    ) -> Dict[str, Any]:
        try:
            eng = make_engine("runs", econ, T=T, ROM_cap=ROM_cap, backend=backend)
            r_star, info = eng.build_and_solve(
                panels, arcs, scenarios or [],
                r_lo=r_lo, r_hi=r_hi, tol=tol, progress_cb=progress_cb
            )
            info_out: Dict[str, Any] = {"ok": True, "r": r_star}
            info_out.update(info)
            return info_out
        except Exception as e:
            return {"ok": False, "error": str(e)}