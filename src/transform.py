"""
transform.py

PySpark transformations: cleans, standardises, and joins supply chain data.
"""

import logging
from typing import Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, DateType

logger = logging.getLogger(__name__)


def get_spark(config: dict) -> SparkSession:
    return (
        SparkSession.builder
        .appName(config.get("app_name", "SupplyChainETL"))
        .master(config.get("master", "local[*]"))
        .config("spark.executor.memory", config.get("executor_memory", "4g"))
        .config("spark.driver.memory", config.get("driver_memory", "2g"))
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def clean_inventory(df: DataFrame) -> DataFrame:
    """Standardise inventory records."""
    return (
        df
        .withColumnRenamed("SKU", "sku")
        .withColumnRenamed("StockQty", "stock_quantity")
        .withColumn("stock_quantity", F.col("stock_quantity").cast(IntegerType()))
        .withColumn("unit_cost", F.col("unit_cost").cast(DoubleType()))
        .withColumn("last_updated", F.to_date(F.col("last_updated"), "yyyy-MM-dd"))
        .filter(F.col("sku").isNotNull())
        .filter(F.col("stock_quantity") >= 0)
        .dropDuplicates(["sku"])
    )


def clean_orders(df: DataFrame) -> DataFrame:
    """Standardise order records."""
    return (
        df
        .withColumn("order_date", F.to_date(F.col("order_date"), "yyyy-MM-dd"))
        .withColumn("units_ordered", F.col("units_ordered").cast(IntegerType()))
        .withColumn("units_sold", F.col("units_sold").cast(IntegerType()))
        .filter(F.col("sku").isNotNull())
        .filter(F.col("units_ordered") > 0)
        .na.fill({"units_sold": 0})
    )


def join_supply_chain(inventory: DataFrame, orders: DataFrame,
                      suppliers: DataFrame) -> DataFrame:
    """
    Join inventory, orders, and supplier data into a unified supply chain table.
    """
    orders_agg = (
        orders
        .groupBy("sku", "order_date")
        .agg(
            F.sum("units_ordered").alias("total_ordered"),
            F.sum("units_sold").alias("total_sold"),
            F.countDistinct("order_id").alias("order_count"),
        )
    )

    joined = (
        orders_agg
        .join(inventory.select("sku", "stock_quantity", "unit_cost", "category"), on="sku", how="left")
        .join(suppliers.select("sku", "supplier_name", "lead_time_days"), on="sku", how="left")
        .withColumn("stockout_flag", F.when(F.col("stock_quantity") == 0, 1).otherwise(0))
        .withColumn("days_of_supply",
                    F.when(F.col("total_sold") > 0,
                           (F.col("stock_quantity") / F.col("total_sold")).cast(IntegerType()))
                    .otherwise(None))
    )

    logger.info("Joined supply chain table: %d rows", joined.count())
    return joined


def transform_all(raw_data: Dict[str, DataFrame]) -> DataFrame:
    inv = clean_inventory(raw_data["inventory"])
    orders = clean_orders(raw_data["orders"])
    suppliers = raw_data["suppliers"]
    return join_supply_chain(inv, orders, suppliers)
