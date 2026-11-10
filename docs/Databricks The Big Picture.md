# Databricks: The Big Picture

Nov 9, 2026 · @Ziad Farid

## The whole platform in one view

&#91;embedded content: Databricks platform · how the parts connect\]

You write code in the workspace, compute runs it, and every table or file it reads or writes is registered in Unity Catalog, while the actual files sit in cloud storage as Delta.

## Workspace and notebooks

The workspace is the browser app where you do everything: write code, browse data, schedule jobs. It is the same on any laptop because nothing runs locally.

- **Notebooks** hold cells of Python, SQL or Markdown. The first line of a cell (`%sql`, `%md`, `%python`) sets its language.
- **Built-in objects:** `spark` (the session), `dbutils` (files, secrets, widgets) and `display()` exist without imports.
- **Git folders** link a workspace folder to a GitHub repo, so you commit and push from the browser.
- **SQL editor and dashboards** let you query tables and build charts without a notebook.

## Compute

Compute is the machines that actually run your code; the workspace only sends it instructions. Spark splits each job across many machines and reassembles the result.

| Type | What it is | When it's used |
| --- | --- | --- |
| Serverless | Databricks starts and manages the machines for you, in seconds | Your Free Edition notebooks; the default direction for new work |
| Classic (all-purpose) cluster | Machines you configure: size, Spark version, libraries | Interactive work in company workspaces |
| Job cluster | A cluster created for one scheduled run, then shut down | Production pipelines (cheaper than leaving one running) |
| SQL warehouse | Compute tuned for SQL queries and dashboards | Analysts and BI tools like Power BI |

SQL Server comparison: it's as if the database engine were rented by the minute and could add servers to a slow query.

## Unity Catalog

Unity Catalog is the single place that knows where every table and file is and who can touch it. Every object has a three-part name: `catalog.schema.object`.

| Level | SQL Server equivalent | Your project |
| --- | --- | --- |
| Metastore | The instance | Hidden, one per region |
| Catalog | A database | `workspace` |
| Schema | A schema (`dbo`) | `bronze`, `silver`, later `gold` |
| Table, view, volume, function | Table, view, file share, function | `fhir_bundles_raw`, `raw_files`, `patients` |

- **Permissions flow down.** Reading a table needs `USE CATALOG`, `USE SCHEMA` and `SELECT`. A grant on a schema covers everything inside it.
- **Catalogs isolate environments or domains** (`dev` / `prod`, `finance` / `clinical`). Code stays the same; only the catalog name changes.
- **Volumes** store files (JSON, CSV, images) under the same permissions as tables.
- **Managed vs external (exam):** a managed table's files sit in storage Databricks controls, and dropping the table deletes them. An external table points at storage you control, and dropping it removes only the definition.
- **Lineage and audit** come for free: Unity Catalog records which table was built from which.

## Storage and Delta Lake

All data physically lives as files in cheap cloud object storage (S3, Azure Blob, Google Cloud Storage). Delta Lake is the format that makes those files behave like a real database table.

A Delta table is two things in one folder:

1. **Parquet files** holding the data, stored column by column and compressed.
2. **A transaction log** (`_delta_log`) recording every change as a numbered version.

The log is what gives you database behavior on plain files:

- **ACID transactions:** a write fully succeeds or leaves no trace, like `BEGIN TRAN` / `COMMIT`.
- **Time travel (exam):** `SELECT * FROM t VERSION AS OF 3` reads an old version. `DESCRIBE HISTORY t` lists them all, and `RESTORE` rolls back.
- **MERGE INTO (exam):** upserts, the same idea as T-SQL `MERGE`. This is how silver and gold tables usually stay up to date.
- **Schema enforcement:** a write with the wrong columns fails instead of corrupting the table, unless you allow evolution (`mergeSchema`).
- **Maintenance:** `OPTIMIZE` compacts many small files into fewer large ones; `VACUUM` deletes old files no longer needed (which also limits how far back time travel can go).

## Medallion architecture

Data moves through three layers, each one cleaner and closer to what the business asks for. Each layer is a schema of Delta tables.

| Layer | What it holds | Rules | Your project |
| --- | --- | --- | --- |
| Bronze | Raw data exactly as received | No cleaning; add lineage columns (source file, load time); never lose anything | `fhir_bundles_raw`: 114 files as text |
| Silver | Cleaned, typed, deduplicated tables, one per business entity | Fix types, drop duplicates, apply quality checks, keep detail | `patients`, then encounters, conditions, medications |
| Gold | Tables shaped for a question: star schemas, aggregates | Modeled for BI and reporting | Patient and encounter facts and dimensions, dashboard |

SQL Server comparison: bronze is your staging tables, silver is the cleaned operational tables, gold is the reporting views your FastReport and Power BI work reads from.

## Ingestion and orchestration

Ingestion gets data into bronze; orchestration runs the whole pipeline on a schedule, in the right order. Databricks groups these tools under the name Lakeflow.

| Tool | What it does | SQL Server equivalent |
| --- | --- | --- |
| `spark.read` (batch) | Reads all files every time | A one-off `BULK INSERT` |
| `COPY INTO` (exam) | SQL command that loads only files it hasn't loaded before | `BULK INSERT` with a log of processed files |
| Auto Loader (exam) | Watches a folder and loads only new files, with checkpoints and schema evolution | A file-watcher job plus a "processed files" table, built in |
| Lakeflow Connect | Managed connectors pulling from databases and SaaS apps | Linked servers and replication |
| Lakeflow Declarative Pipelines (exam) | You declare tables and quality rules; Databricks works out the order and runs them. Formerly called Delta Live Tables | Nothing direct; closest is SSIS with built-in data checks |
| Lakeflow Jobs (exam) | Schedules notebooks and pipelines as tasks with dependencies, retries and alerts. Formerly called Workflows | SQL Server Agent jobs with steps |

The key exam idea is incremental processing: load only what is new since the last run, instead of reprocessing everything. Your bronze notebook currently reloads all files on every run; switching it to Auto Loader is a natural upgrade later.

## Where your project sits

After day one you have already touched about half of the platform. Tick items off as the project reaches them.

- [x] Workspace, notebooks, magic commands
- [x] Serverless compute
- [x] Unity Catalog: catalog, schemas, a volume
- [x] Git folder linked to GitHub
- [x] Bronze: raw files into a Delta table with lineage columns
- [ ] Silver: patients (SQL done, saving and PySpark next), then encounters, conditions, medications
- [ ] Delta features in practice: `DESCRIBE HISTORY`, time travel, `MERGE INTO`
- [ ] Auto Loader for incremental bronze
- [ ] Gold: star schema and a dashboard
- [ ] Lakeflow Jobs: schedule bronze to silver to gold as one pipeline
- [ ] Permissions: a `GRANT` on the gold schema

## Glossary

| Databricks term | Closest SQL Server idea |
| --- | --- |
| Lakehouse | Data warehouse and file share in one, on cheap storage |
| Spark | A query engine that splits work across many machines |
| DataFrame | A query result you keep transforming, like a chain of CTEs |
| Transformation vs action | Building a query plan vs executing it |
| Delta table | A table with a full change log and version history |
| Managed table | Table whose files the engine owns and deletes on `DROP` |
| External table | Table defined over files you control |
| Volume | A governed file share inside a schema |
| Catalog | Database |
| Checkpoint | A bookmark of what has already been processed |
| Job / task | Agent job / job step |
| Pipeline | A set of dependent loads run together, like an SSIS package |
| Photon | A faster execution engine Databricks switches on for you |
