# Healthcare Lakehouse on Databricks

End-to-end pipeline that turns raw HL7 FHIR R4 patient bundles into an analytics-ready star schema, using a medallion architecture on Databricks.

## Problem
Clinical data arrives as deeply nested FHIR JSON (one bundle per patient, dozens of resource types). Analysts need flat, trustworthy tables: patients, encounters, conditions, medications.

## Architecture
`raw FHIR JSON → Bronze (raw, one row per file) → Silver (parsed, typed, deduplicated) → Gold (star schema) → Dashboard`

_Diagram: coming with the silver layer._

## Stack
Databricks (Free Edition) · PySpark · Spark SQL · Delta Lake · Unity Catalog (volumes, schemas) · Lakeflow Jobs (planned)

## Data
Synthetic patient records from [Synthea](https://synthea.mitre.org/downloads) (public domain, no real patient data).

## Layers
| Layer | Status | Notebook |
|---|---|---|
| Bronze | ✅ Done | `01_bronze_fhir_raw.py` |
| Silver | ⏳ | |
| Gold | ⏳ | |
| Orchestration | ⏳ | |

## How to run
1. Create a Databricks Free Edition workspace.
2. Import `01_bronze_fhir_raw.py` as a notebook and run the first cell.
3. Upload Synthea FHIR R4 JSON files to the volume path it prints.
4. Run the remaining cells.

## Design decisions
- **Bronze stores raw text, not parsed JSON.** FHIR bundles have inconsistent nested schemas; keeping raw text means nothing is lost and parsing logic can change without re-ingesting.
- **Bronze writes are idempotent** (overwrite), so re-runs never create duplicates.

## What I learned
_(fill in as you go)_
