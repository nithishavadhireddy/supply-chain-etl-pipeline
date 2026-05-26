"""
extract.py

Reads data from multiple source systems (CSV files, databases, REST APIs)
and returns raw DataFrames for the transformation layer.
"""

import logging
import os
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def extract_csv(path: str, source_name: str) -> pd.DataFrame:
    """Load a CSV source file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Extracted %d rows from %s (%s)", len(df), source_name, path)
    return df


def extract_database(conn_string: str, query: str, source_name: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    from sqlalchemy import create_engine, text
    engine = create_engine(conn_string)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    logger.info("Extracted %d rows from DB source: %s", len(df), source_name)
    return df


def extract_rest_api(url: str, headers: dict = None, params: dict = None,
                     source_name: str = "api") -> pd.DataFrame:
    """Fetch JSON data from a REST API endpoint."""
    import requests
    resp = requests.get(url, headers=headers or {}, params=params or {}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Handle both list and {"data": [...]} envelope formats
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif "data" in data:
        df = pd.DataFrame(data["data"])
    else:
        df = pd.DataFrame([data])
    logger.info("Extracted %d rows from API source: %s", len(df), source_name)
    return df


def extract_all(config: dict) -> Dict[str, pd.DataFrame]:
    """
    Extract from all configured sources. Returns a dict keyed by source name.
    """
    sources = config.get("sources", {})
    result = {}

    for name, cfg in sources.items():
        src_type = cfg.get("type", "csv")
        if src_type == "csv":
            result[name] = extract_csv(cfg["path"], name)
        elif src_type == "database":
            result[name] = extract_database(cfg["connection"], cfg["query"], name)
        elif src_type == "api":
            result[name] = extract_rest_api(cfg["url"], cfg.get("headers"), cfg.get("params"), name)
        else:
            logger.warning("Unknown source type '%s' for source '%s', skipping", src_type, name)

    logger.info("Extraction complete. Sources loaded: %s", list(result.keys()))
    return result
