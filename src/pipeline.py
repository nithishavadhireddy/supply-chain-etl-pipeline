"""
pipeline.py

End-to-end supply chain ETL pipeline runner.

Usage:
    python src/pipeline.py --config config/pipeline_config.yaml
"""

import argparse
import logging
import os

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Starting supply chain ETL pipeline")

    # 1. Extract
    from extract import extract_all
    raw_data = extract_all(config)

    # 2. Transform with PySpark
    from transform import get_spark, transform_all
    spark = get_spark(config.get("spark", {}))

    spark_dfs = {
        name: spark.createDataFrame(pdf)
        for name, pdf in raw_data.items()
    }
    transformed = transform_all(spark_dfs)

    # 3. Load to S3
    from load import save_to_s3_parquet
    aws = config["aws"]
    s3_path = save_to_s3_parquet(
        transformed,
        s3_bucket=aws["s3_bucket"],
        s3_key=f"{aws['s3_processed_prefix']}/supply_chain_unified",
    )

    # 4. Load to Redshift (via COPY)
    from load import redshift_copy_from_s3
    redshift_conn = (
        f"postgresql+psycopg2://{os.environ['REDSHIFT_USER']}:{os.environ['REDSHIFT_PASSWORD']}"
        f"@{aws['redshift_host']}:{aws['redshift_port']}/{aws['redshift_db']}"
    )
    redshift_copy_from_s3(
        s3_path=s3_path,
        table="supply_chain_unified",
        schema=aws["redshift_schema"],
        redshift_conn_str=redshift_conn,
        iam_role=os.environ["REDSHIFT_IAM_ROLE"],
    )

    # 5. Forecast
    from forecast import train_forecast_model, generate_sku_forecasts
    pandas_df = transformed.toPandas()
    forecast_config = config.get("forecast", {})
    model, metrics = train_forecast_model(pandas_df, forecast_config)
    logger.info("Forecast metrics: %s", metrics)

    forecasts = generate_sku_forecasts(model, pandas_df, forecast_config.get("horizon_days", 30))
    logger.info("Top 5 SKUs by forecasted demand:\n%s", forecasts.head().to_string(index=False))

    spark.stop()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply chain ETL pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()
    main(args.config)
