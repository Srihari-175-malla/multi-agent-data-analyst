"""
Loads CSV/Excel uploads into an in-process DuckDB database so the SQL Agent
can query them directly, and exposes the same data as a pandas DataFrame for
the Python Agent and Statistics Agent.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

from src.config import settings
from src.utils_logging import get_logger

logger = get_logger(__name__)


@dataclass
class Dataset:
    dataset_id: str
    table_name: str
    df: pd.DataFrame
    schema: Dict[str, str]
    source_path: str
    con: duckdb.DuckDBPyConnection


class DataLoader:
    """Keeps one DuckDB connection + DataFrame per loaded dataset, in memory."""

    _datasets: Dict[str, Dataset] = {}

    @classmethod
    def load(cls, file_path: str, dataset_id: str = None) -> Dataset:
        file_path = str(file_path)
        dataset_id = dataset_id or Path(file_path).stem
        table_name = cls._sanitize_table_name(dataset_id)

        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        df = cls._normalize_columns(df)

        con = duckdb.connect(database=":memory:")
        con.register(table_name, df)

        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

        dataset = Dataset(
            dataset_id=dataset_id,
            table_name=table_name,
            df=df,
            schema=schema,
            source_path=file_path,
            con=con,
        )
        cls._datasets[dataset_id] = dataset
        logger.info(f"Loaded dataset '{dataset_id}' -> table '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")
        return dataset

    @classmethod
    def get(cls, dataset_id: str) -> Dataset:
        if dataset_id not in cls._datasets:
            raise KeyError(f"Dataset '{dataset_id}' not loaded. Upload it first.")
        return cls._datasets[dataset_id]

    @staticmethod
    def _sanitize_table_name(name: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in name)
        if safe and safe[0].isdigit():
            safe = f"t_{safe}"
        return safe or "data"

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
        # try to parse an obvious date column
        for col in df.columns:
            if "date" in col or "time" in col:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
        return df
