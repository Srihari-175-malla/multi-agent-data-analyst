"""
Convenience charting tool: standard chart types (line/bar/scatter) without
requiring the agent to hand-write matplotlib code. Runs in-process since
plotting from fixed inputs (column names) carries none of the arbitrary-code
risk that free-form Python execution does.
"""
import uuid
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings


class PlottingTool:
    def __init__(self, df: pd.DataFrame, chart_dir: str = None):
        self.df = df
        self.chart_dir = Path(chart_dir or settings.charts_dir)
        self.chart_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, chart_type: str, x: str, y: str, group_by: Optional[str] = None, title: str = "") -> str:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        if chart_type == "line":
            if group_by:
                for key, sub in self.df.groupby(group_by):
                    ax.plot(sub[x], sub[y], label=str(key))
                ax.legend()
            else:
                ax.plot(self.df[x], self.df[y])
        elif chart_type == "bar":
            grouped = self.df.groupby(x)[y].mean() if group_by is None else self.df.groupby([x, group_by])[y].mean().unstack()
            grouped.plot(kind="bar", ax=ax)
        elif chart_type == "scatter":
            ax.scatter(self.df[x], self.df[y], alpha=0.6)
        else:
            raise ValueError(f"Unsupported chart_type: {chart_type}")

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f"{y} by {x}")
        fig.autofmt_xdate()

        path = self.chart_dir / f"{chart_type}_{uuid.uuid4().hex[:8]}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return str(path)
