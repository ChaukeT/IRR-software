from __future__ import annotations
from pathlib import Path
import pandas as pd
from irr_core.modelio import load_panels_from_df


def load_blockmodel_csv(path: Path):
	df = pd.read_csv(path)
	return load_panels_from_df(df)