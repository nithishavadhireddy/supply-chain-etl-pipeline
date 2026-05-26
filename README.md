# supply-chain-etl-pipeline

End-to-end ETL pipeline for pharmaceutical supply chain data. Consolidates data from
multiple source systems into a centralised S3 data lake, processes it with PySpark,
loads to Redshift, and runs a demand forecasting model for inventory planning.

Inspired by work at a pharma company where supply chain data was siloed across
5+ source systems, making it nearly impossible for teams to get a single view of operations.

## Architecture

```
Source Systems (CSV / DB / API)
        ↓
    extract.py   (pull from each source)
        ↓
    transform.py (PySpark cleaning + standardisation)
        ↓
    load.py      (S3 data lake + Redshift COPY)
        ↓
    forecast.py  (demand forecasting with Random Forest)
        ↓
    pipeline.py  (orchestrates the full run)
```

## Tech Stack

- **Processing**: PySpark 3.5
- **Storage**: AWS S3 (data lake), AWS Redshift (warehouse)
- **Forecasting**: scikit-learn Random Forest with feature engineering
- **Orchestration**: pipeline.py (can be triggered from Airflow or CLI)

## Setup

```bash
git clone https://github.com/<username>/supply-chain-etl-pipeline
cd supply-chain-etl-pipeline
pip install -r requirements.txt

cp .env.example .env
# Fill in AWS credentials, Redshift connection

# Run the full pipeline
python src/pipeline.py --config config/pipeline_config.yaml
```

## Project Structure

```
supply-chain-etl-pipeline/
├── src/
│   ├── extract.py        # Source connectors
│   ├── transform.py      # PySpark transformations
│   ├── load.py           # S3 + Redshift loader
│   ├── forecast.py       # Demand forecasting
│   └── pipeline.py       # End-to-end orchestrator
└── config/
    └── pipeline_config.yaml
```

## Demand Forecasting

The forecasting module trains a Random Forest regressor on historical stock
and order data to predict demand for the next N days per SKU. Features include:
- Rolling mean / std of historical demand
- Day-of-week and month seasonality
- Promotional flag
- Stockout indicator from prior period
