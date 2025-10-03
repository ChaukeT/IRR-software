# IRR Scheduler — PySide6


## Install
```bash
python -m venv .venv
. .venv/Scripts/activate # Windows
# or: source .venv/bin/activate
pip install -r requirements.txt
```


## Run
```bash
python -m app.main
```


## Usage
1. Load your blockmodel CSV (must match columns expected by irr_core.modelio.load_panels_from_df).
2. Choose backend = `local` (no external solver) or `external`.
3. Set horizon, ROM cap, and IRR search bounds.
4. Run. IRR* and a mined schedule table will be shown if feasible.