"""
load.py

Loads processed DataFrames to S3 (as Parquet) and copies into Redshift.
"""

import logging
import os

import boto3
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


def save_to_s3_parquet(df, s3_bucket: str, s3_key: str, aws_region: str = "us-east-1"):
    """Write a Spark DataFrame to S3 as Parquet."""
    s3_path = f"s3a://{s3_bucket}/{s3_key}"
    (
        df.write
        .mode("overwrite")
        .parquet(s3_path)
    )
    logger.info("Saved to S3: %s", s3_path)
    return s3_path


def redshift_copy_from_s3(s3_path: str, table: str, schema: str,
                           redshift_conn_str: str, iam_role: str):
    """
    Run a Redshift COPY command to load Parquet data from S3.
    Much faster than row-by-row inserts for large datasets.
    """
    copy_sql = f"""
        COPY {schema}.{table}
        FROM '{s3_path}'
        IAM_ROLE '{iam_role}'
        FORMAT AS PARQUET;
    """
    engine = create_engine(redshift_conn_str)
    with engine.connect() as conn:
        conn.execute(copy_sql)
        conn.execute("COMMIT")
    logger.info("Redshift COPY complete: %s.%s", schema, table)


def load_pandas_to_redshift(df: pd.DataFrame, table: str, schema: str,
                             conn_str: str, if_exists: str = "append"):
    """Fallback: use pandas to_sql for smaller DataFrames (dev / testing)."""
    engine = create_engine(conn_str)
    df.to_sql(table, engine, schema=schema, if_exists=if_exists, index=False, method="multi")
    logger.info("Loaded %d rows to %s.%s via pandas", len(df), schema, table)
