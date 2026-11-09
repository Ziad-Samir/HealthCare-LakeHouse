# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Bronze: raw FHIR bundles
# MAGIC Lands Synthea FHIR R4 bundles exactly as received: one row per file, no parsing.
# MAGIC Parsing into patients / encounters / medications happens in silver.

# COMMAND ----------

# Free Edition's default catalog is usually "workspace"; check Catalog Explorer and change if needed
CATALOG = "workspace"
SCHEMA = "bronze"
VOLUME = "raw_files"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

LANDING = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/fhir"
TARGET = f"{CATALOG}.{SCHEMA}.fhir_bundles_raw"
print("Upload JSON files to:", LANDING)

# COMMAND ----------

# MAGIC %md
# MAGIC **Before running the next cell:** Catalog → workspace → bronze → raw_files → create folder `fhir` → upload ~100 JSON files.

# COMMAND ----------

from pyspark.sql import functions as F

raw = (
    spark.read.option("wholetext", True).text(f"{LANDING}/*.json")
    .select(
        F.col("value").alias("raw_json"),
        F.col("_metadata.file_path").alias("source_file"),
        F.current_timestamp().alias("ingested_at"),
    )
)

# overwrite = idempotent: re-running gives the same table, never duplicates
raw.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(TARGET)

# COMMAND ----------

# Sanity check: every file should be a Bundle; entries = FHIR resources inside it
display(spark.sql(f"""
    SELECT get_json_object(raw_json, '$.resourceType')                       AS resource_type,
           COUNT(*)                                                          AS files,
           SUM(json_array_length(get_json_object(raw_json, '$.entry')))      AS total_resources,
           ROUND(SUM(length(raw_json)) / 1024 / 1024, 1)                     AS size_mb
    FROM {TARGET}
    GROUP BY 1
"""))

# COMMAND ----------

# Which resource types are inside the bundles? (this is the silver-layer to-do list)
display(spark.sql(f"""
    SELECT get_json_object(e, '$.resource.resourceType') AS resource_type, COUNT(*) AS n
    FROM {TARGET}
    LATERAL VIEW explode(from_json(get_json_object(raw_json, '$.entry'), 'array<string>')) t AS e
    GROUP BY 1
    ORDER BY n DESC
"""))