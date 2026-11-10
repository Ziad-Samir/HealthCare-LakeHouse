# PySpark Cheat Sheet for T-SQL Developers
*Databricks serverless · Unity Catalog · Data Engineer Associate prep*

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
```

**Legend:** **(exam)** = high value for the Databricks Certified Data Engineer Associate · 🆕 = new or changed in recent Spark / Databricks versions · ⚠️ = gotcha

> Version notes reflect Spark 3.5 / 4.0 and Databricks serverless as of late 2026. Serverless environment versions move fast, so when a 🆕 item matters, confirm it in the docs for the environment version your notebook uses.

## Contents
0. [Blank-cell starter](#0-blank-cell-starter)
1. [Core mental model](#1-core-mental-model)
2. [Reading data](#2-reading-data)
3. [Inspecting data](#3-inspecting-data)
4. [Selecting and shaping](#4-selecting-and-shaping)
5. [Filtering](#5-filtering)
6. [Functions by category](#6-functions-by-category)
7. [Aggregations](#7-aggregations)
8. [Joins](#8-joins)
9. [Window functions](#9-window-functions)
10. [Complex and nested data](#10-complex-and-nested-data)
11. [Set operations and dedup](#11-set-operations-and-dedup)
12. [Writing data](#12-writing-data)
13. [Delta Lake essentials (exam)](#13-delta-lake-essentials-exam)
14. [Mixing SQL and Python](#14-mixing-sql-and-python)
15. [Common errors and gotchas](#15-common-errors-and-gotchas)
16. [Performance basics](#16-performance-basics)
17. [Most used 20](#most-used-20)

---

## 0. Blank-cell starter

When you're stuck on an empty cell, write the query in T-SQL's **logical processing order**. PySpark methods map almost one-to-one, in that order:

| T-SQL (logical order) | PySpark |
|---|---|
| `FROM t` | `spark.table("cat.sch.t")` |
| `JOIN` | `.join(other, "key", "left")` |
| `WHERE` | `.filter(...)` |
| `GROUP BY` + aggregates | `.groupBy(...).agg(...)` |
| `HAVING` | `.filter(...)` after `.agg()` |
| `SELECT` list / computed columns | `.select(...)` / `.withColumn(...)` |
| `DISTINCT` | `.distinct()` |
| `ORDER BY` | `.orderBy(...)` |
| `TOP n` | `.limit(n)` |
| `INSERT INTO` / `SELECT INTO` | `.write.mode(...).saveAsTable(...)` |

```python
src = "/Volumes/main/raw/landing/orders/"
tgt = "main.silver.orders"

df = (spark.read
      .format("csv")
      .option("header", True)
      .schema("order_id INT, customer_id INT, order_ts TIMESTAMP, amount DECIMAL(18,2), status STRING")
      .load(src))

out = (df
       .filter(F.col("status") != "cancelled")
       .withColumn("order_date", F.to_date("order_ts"))
       .select("order_id", "customer_id", "order_date", "amount"))

out.write.mode("overwrite").saveAsTable(tgt)
display(spark.table(tgt))
```

---

## 🆕 What changed recently (read once)

| Change | What it means for you |
|---|---|
| Serverless runs on **Spark Connect** | No `spark.sparkContext`, no RDD API (`df.rdd`, `sc.parallelize`). Errors often surface only when an action runs, not on the line that caused them. |
| `cache()` / `persist()` not supported on serverless | Materialize intermediate results to a Delta table instead (like a `#temp` table). |
| **ANSI SQL mode** | Default in Spark 4.0 and Databricks SQL. Bad casts, overflow and divide-by-zero **raise errors** instead of returning NULL. Check yours: `spark.conf.get("spark.sql.ansi.enabled")`. Use `try_cast`, `try_divide`, `try_to_timestamp`. |
| `input_file_name()` deprecated | Use the `_metadata` column. Not supported in Unity Catalog. |
| `spark.conf.set(...)` restricted on serverless | Only a small allowlist of configs can be set; adaptive query execution (AQE) handles shuffle partitions. |
| **Liquid clustering** | Recommended over `partitionBy` + `ZORDER` for new tables. (exam) |
| DLT renamed **Lakeflow Declarative Pipelines** (2025) | Same concepts (streaming tables, materialized views, expectations). Exam material may use either name. (exam) |
| New helpers | `withColumns`, `withColumnsRenamed`, `unpivot`, `count_if`, `max_by`/`min_by`, `try_*` functions, parameterized `spark.sql`, `QUALIFY`, SQL variables, `EXECUTE IMMEDIATE`, `VARIANT` type, collations. |

---

## 1. Core mental model

### DataFrames vs tables

| Concept | T-SQL analogy | Notes |
|---|---|---|
| DataFrame | A view definition / a query you haven't executed | Holds a *plan*, not data |
| Unity Catalog table `catalog.schema.table` | `database.schema.table` | Persisted Delta table, three-level namespace (exam) |
| Temp view | A view that only exists in your session | Recomputed every time it's queried |

### Transformations vs actions, lazy evaluation (exam)

| Transformations: lazy, return a new DataFrame | Actions: trigger execution |
|---|---|
| `select`, `filter`/`where`, `withColumn`, `join`, `groupBy().agg()`, `orderBy`, `distinct`, `union`, `limit` | `show`, `display`, `count`, `collect`, `take`/`head`/`first`, `toPandas`, `write...`/`saveAsTable` |

```python
paid = df.filter(F.col("status") == "paid")   # nothing runs yet, Spark just records the plan
paid.count()                                    # NOW Spark reads, filters and counts
```
⚠️ Every action re-runs the whole plan from the source. `count()` followed by `display()` = two full reads.

### Immutability and chaining

```python
df.withColumn("x", F.lit(1))        # ⚠️ result is thrown away, df is unchanged
df = df.withColumn("x", F.lit(1))   # ✅ reassign

result = (
    spark.table("main.sales.orders")
    .filter(F.col("status") == "paid")
    .groupBy("customer_id")
    .agg(F.sum("amount").alias("total"))
)
```
⚠️ Wrap multi-line chains in parentheses instead of using `\` line continuations.

### `F.col` vs plain strings

| Situation | Example |
|---|---|
| Plain string is fine when you only *name* a column | `df.select("a", "b")`, `groupBy("k")`, `drop("x")`, `F.sum("amount")` |
| `F.col()` is needed to build an *expression* | `F.col("amount") * 1.14`, `F.col("a") > 5`, `.alias()`, `.cast()`, `.desc()` |
| `df["col"]` / `df.col` | Bound to that specific DataFrame; use it to disambiguate after joins |
| `F.lit()` turns a Python value into a column | `F.concat("first_name", F.lit(" "), "last_name")` |

⚠️ `"amount" * 2` is Python string repetition (`"amountamount"`), not math.
⚠️ Inside functions a string usually means a column **name**. Exceptions are *value* arguments: the 2nd arg of `F.when(cond, "Y")`, `fillna("x")`, `isin("a", "b")`, where strings are literals.

### Databricks built-ins: `spark`, `dbutils`, `display`

| Object | What it is | Typical use |
|---|---|---|
| `spark` | Pre-created SparkSession | `spark.table()`, `spark.read`, `spark.sql()` |
| `dbutils` | Databricks utilities | files, widgets (parameters), secrets, task values |
| `display(df)` | Rich sortable grid + charts | Notebooks only; use `df.show()` in plain `.py` scripts |

```python
dbutils.fs.ls("/Volumes/main/raw/landing/")              # list files
dbutils.fs.cp(src_path, dst_path)                        # copy
dbutils.fs.rm("/Volumes/main/raw/tmp/", recurse=True)    # delete folder
dbutils.widgets.text("run_date", "2026-11-01")           # notebook / job parameter
run_date = dbutils.widgets.get("run_date")               # ⚠️ always returns a string
pwd = dbutils.secrets.get(scope="kv", key="sql-pwd")     # redacted if printed
dbutils.jobs.taskValues.set(key="row_count", value=123)  # pass values between job tasks (exam)
```

---

## 2. Reading data

### Generic pattern

```python
df = (spark.read
      .format("csv")            # csv | json | parquet | delta | text | binaryFile
      .option("header", True)
      .schema(my_schema)        # or .option("inferSchema", True)
      .load("/Volumes/main/raw/landing/orders/"))   # a file, folder, or glob like *.csv
```
Shortcuts: `spark.read.csv(path, header=True, schema=s)`, `spark.read.json(path)`, `spark.read.parquet(path)`, `spark.read.text(path)`.

| T-SQL | Spark SQL |
|---|---|
| `SELECT * FROM OPENROWSET(BULK 'C:\f.csv', ...)` / `BULK INSERT` | `SELECT * FROM read_files('/Volumes/main/raw/landing/orders/', format => 'csv', header => true)` (exam) |
| | Direct file query: ``SELECT * FROM json.`/Volumes/main/raw/landing/events/` `` (exam) |

### Reading from Unity Catalog volumes (exam)

Path format: `/Volumes/<catalog>/<schema>/<volume>/<folder>/<file>`

Volumes hold **files** (CSV, JSON, images, PDFs). **Tables** hold tabular Delta data.
⚠️ In Unity Catalog, use volumes rather than the DBFS root or `/mnt` mounts (legacy).

### CSV

```python
df = (spark.read.format("csv")
      .option("header", True)
      .option("sep", ",")                    # ";" "\t" "|"
      .option("quote", '"')
      .option("escape", '"')                 # Excel-style "" inside quoted fields
      .option("multiLine", True)             # quoted fields that contain line breaks
      .option("nullValue", "NULL")
      .option("dateFormat", "dd/MM/yyyy")
      .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
      .option("encoding", "UTF-8")
      .option("mode", "PERMISSIVE")          # PERMISSIVE (default) | DROPMALFORMED | FAILFAST
      .schema("order_id INT, order_date DATE, amount DECIMAL(18,2), note STRING")
      .load("/Volumes/main/raw/landing/orders.csv"))
```
⚠️ No schema and no `inferSchema` → every column is STRING. ⚠️ `inferSchema` costs an extra pass over the data.
Databricks: `.option("rescuedDataColumn", "_rescued_data")` keeps values that don't fit the schema instead of silently nulling them.

### JSON: single-line vs multiLine

| File shape | Read with |
|---|---|
| JSON Lines, one object per line: `{"id":1}` newline `{"id":2}` | `spark.read.json(path)` (default) |
| Pretty-printed object, or one big array `[{...},{...}]` | `spark.read.option("multiLine", True).json(path)` |

⚠️ Reading a pretty-printed file without `multiLine` gives a single `_corrupt_record` column.
⚠️ A multiLine file can't be split, so one file = one task (slow for one huge file).

### Parquet and Delta

```python
spark.read.parquet("/Volumes/main/raw/landing/parquet/")
spark.read.table("main.silver.orders")           # ✅ preferred for Delta: read by name
spark.table("main.silver.orders")                # same thing
spark.read.format("delta").load("abfss://...")   # by path: needs external-location permission
```
⚠️ In Unity Catalog, read tables by name. Path-based access bypasses table permissions and is restricted.

### Text (including wholetext)

```python
spark.read.text(path)                                            # one row per LINE, column `value`
spark.read.text(path, wholetext=True)                            # one row per FILE
spark.read.format("text").option("wholetext", True).load(path)   # same, generic form
spark.read.format("binaryFile").load(path)                       # path, modificationTime, length, content
```
⚠️ `spark.read.option("wholetext", True).text(path)` **silently reads line by line**: `.text()` has its own `wholetext=False` default that overwrites your option (see §15).

### Schema inference vs explicit schema

| | Inference | Explicit |
|---|---|---|
| Code | `.option("inferSchema", True)` | `.schema(...)` |
| Speed | Extra pass over data | No extra pass |
| Safety | Types can change between files or runs | Stable; the production default |

```python
# DDL string: closest to CREATE TABLE, easiest to write
schema = "order_id INT, customer_id BIGINT, order_ts TIMESTAMP, amount DECIMAL(18,2), status STRING"

# StructType: use when building schemas programmatically
schema = StructType([
    StructField("order_id", IntegerType()),
    StructField("customer_id", LongType()),
    StructField("order_ts", TimestampType()),
    StructField("amount", DecimalType(18, 2)),
    StructField("status", StringType()),
])

# Trick: infer once on a sample, then print a DDL string to paste as an explicit schema
print(", ".join(f"{f.name} {f.dataType.simpleString()}" for f in df.schema.fields))
```
⚠️ `nullable=False` is not enforced when reading files; Spark treats file columns as nullable.

### The `_metadata` column (exam)

```python
df = (spark.read.format("json").load("/Volumes/main/raw/landing/events/")
      .select("*",
              F.col("_metadata.file_path").alias("source_file"),
              F.col("_metadata.file_modification_time").alias("file_mtime")))
```
Fields: `file_path`, `file_name`, `file_size`, `file_modification_time`, `file_block_start`, `file_block_length`.
⚠️ Hidden column: `select("*")` doesn't include it, you must name it. 🆕 Replaces `input_file_name()`.

### Auto Loader basics (exam)

Incrementally ingests **new** files from a folder. Processed files are tracked in the checkpoint, so each file is loaded exactly once.

```python
landing = "/Volumes/main/raw/landing/orders/"
chk     = "/Volumes/main/raw/checkpoints/orders/"

stream = (spark.readStream
          .format("cloudFiles")
          .option("cloudFiles.format", "json")              # csv, parquet, avro, text, binaryFile...
          .option("cloudFiles.schemaLocation", chk)         # where the inferred schema is tracked
          .option("cloudFiles.inferColumnTypes", True)      # otherwise JSON/CSV columns are all STRING
          .option("cloudFiles.schemaEvolutionMode", "addNewColumns")   # default
          .load(landing)
          .withColumn("source_file", F.col("_metadata.file_path"))
          .withColumn("ingested_at", F.current_timestamp()))

(stream.writeStream
   .option("checkpointLocation", chk)
   .option("mergeSchema", True)
   .trigger(availableNow=True)     # process everything new, then stop
   .toTable("main.bronze.orders"))
```

Key facts for the exam:
- **Schema evolution modes:** `addNewColumns` (default: the stream fails when a new column appears and picks it up on restart), `rescue` (unexpected data goes to `_rescued_data`, never fails), `failOnNewColumns`, `none`.
- **`_rescued_data`** captures values that don't match the schema (type mismatches, case mismatches, unexpected fields).
- **Triggers:** `availableNow=True` processes the backlog and stops. ⚠️ `processingTime="1 minute"` (always-on micro-batches) is not supported on serverless compute.
- **Checkpoint** is the record of what was processed. Delete it and everything is reprocessed.
- **Auto Loader vs COPY INTO:** Auto Loader scales to millions of files and supports schema evolution and streaming; COPY INTO is a simpler, idempotent SQL command for thousands of files.

```sql
-- COPY INTO (exam): re-running skips files already loaded
CREATE TABLE IF NOT EXISTS main.bronze.orders;     -- schemaless placeholder table
COPY INTO main.bronze.orders
FROM '/Volumes/main/raw/landing/orders/'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true');

-- Lakeflow Declarative Pipelines (formerly DLT): Auto Loader in SQL (exam)
CREATE OR REFRESH STREAMING TABLE bronze_orders
AS SELECT * FROM STREAM read_files('/Volumes/main/raw/landing/orders/', format => 'json');
```

---

## 3. Inspecting data

| Task | T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|---|
| Look at rows | `SELECT TOP 20 * FROM t` | `df.show()` / `display(df)` | `show` truncates strings at 20 chars: `df.show(20, truncate=False)`; `vertical=True` for wide rows |
| Structure | `EXEC sp_help 't'` | `df.printSchema()` | Shows types, nullability, nesting |
| Column names | `sys.columns` | `df.columns` | A Python list; an attribute, no `()` |
| Names + types | `INFORMATION_SCHEMA.COLUMNS` | `df.dtypes` | List of `(name, type)` tuples |
| Schema object | | `df.schema` | `StructType`, reusable in `.schema()` |
| Row count | `SELECT COUNT(*) FROM t` | `df.count()` | Action: full scan |
| Basic stats | | `df.describe()` / `df.summary()` | Returns a DataFrame of strings; `summary` adds percentiles |
| Limit rows | `TOP 10` | `df.limit(10)` | Transformation (lazy) |
| Rows into Python | | `df.take(5)`, `df.head(5)`, `df.first()` | Action; returns `Row` objects |
| Is it empty? | `IF EXISTS (SELECT 1 FROM t)` | `df.isEmpty()` | 🆕 3.3+; much cheaper than `count() == 0` |
| Table metadata | `EXEC sp_help` | `DESCRIBE EXTENDED main.silver.orders` / `DESCRIBE DETAIL ...` | (exam) |

```python
# SELECT @total = SUM(amount) FROM orders
total = df.agg(F.sum("amount")).first()[0]

row = df.first()
row["amount"], row.amount, row.asDict()    # three ways to read a Row
```

---

## 4. Selecting and shaping

**T-SQL**
```sql
SELECT order_id,
       customer_id          AS cust_id,
       amount * 1.14        AS amount_with_tax,
       CAST(order_ts AS DATE) AS order_date
FROM orders;
```
**PySpark**
```python
df.select(
    "order_id",
    F.col("customer_id").alias("cust_id"),
    (F.col("amount") * 1.14).alias("amount_with_tax"),
    F.col("order_ts").cast("date").alias("order_date"),
)
```
**Spark SQL:** identical to the T-SQL.

| Task | T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|---|
| All + one computed | `SELECT *, a + b AS c` | `df.select("*", (F.col("a") + F.col("b")).alias("c"))` | |
| Add / replace column | computed column | `df.withColumn("c", F.col("a") + F.col("b"))` | Same name = replaces the column |
| Add several | | `df.withColumns({"c": F.col("a") + 1, "src": F.lit("crm")})` | 🆕 3.3+; use instead of `withColumn` in a loop |
| Rename | `sp_rename` / `AS` | `df.withColumnRenamed("old", "new")` | Silent no-op if `old` doesn't exist |
| Rename several | | `df.withColumnsRenamed({"a": "x", "b": "y"})` | 🆕 3.4+ |
| Rename all | | `df.toDF(*[c.lower().replace(" ", "_") for c in df.columns])` | Must pass exactly one name per column |
| Drop | | `df.drop("a", "b")` | No error if a column is missing (typos go unnoticed) |
| Reorder | column list | `df.select("b", "a", *[c for c in df.columns if c not in ("a", "b")])` | |
| Alias | `AS x` | `.alias("x")` | |
| Cast | `CAST(x AS INT)` | `F.col("x").cast("int")` or `.cast(IntegerType())` | Invalid value → NULL (non-ANSI) or **error** (ANSI) |
| Safe cast | `TRY_CAST(x AS INT)` | `F.expr("try_cast(x AS INT)")`; 🆕 4.0: `F.col("x").try_cast("int")` | |
| SQL snippets | | `df.selectExpr("order_id", "amount * 1.14 AS gross", "CAST(order_ts AS DATE) AS d")` | Great when you think in SQL |
| One SQL expression | | `df.withColumn("tier", F.expr("CASE WHEN amount > 100 THEN 'big' ELSE 'small' END"))` | |
| Sort | `ORDER BY a DESC, b` | `df.orderBy(F.desc("a"), "b")` or `F.col("a").desc_nulls_last()` | NULLs first in ASC, last in DESC (same as T-SQL) |

⚠️ You can't reference an alias in the same `select` (same rule as T-SQL); chain another `select`/`withColumn`. 🆕 Databricks **Spark SQL** does allow lateral column aliases: `SELECT a * 2 AS b, b + 1 AS c FROM t`.

### Type mapping

| SQL Server | Spark type | DDL string | ⚠️ Gotcha |
|---|---|---|---|
| `INT` | `IntegerType()` | `int` | |
| `BIGINT` | `LongType()` | `bigint` | |
| `SMALLINT` / `TINYINT` | `ShortType()` / `ByteType()` | `smallint` / `tinyint` | Spark TINYINT is signed (−128..127); SQL Server's is 0..255 |
| `BIT` | `BooleanType()` | `boolean` | |
| `DECIMAL(p,s)` / `MONEY` | `DecimalType(p, s)` | `decimal(18,2)` | |
| `FLOAT` / `REAL` | `DoubleType()` / `FloatType()` | `double` / `float` | |
| `VARCHAR(n)` / `NVARCHAR(n)` | `StringType()` | `string` | No length limit; always Unicode |
| `DATE` | `DateType()` | `date` | |
| `DATETIME` / `DATETIME2` | `TimestampNTZType()` 🆕 or `TimestampType()` | `timestamp_ntz` / `timestamp` | `timestamp` is time-zone aware (stored as UTC, shown in the session time zone); `timestamp_ntz` is wall-clock like `DATETIME2` |
| `DATETIMEOFFSET` | `TimestampType()` | `timestamp` | |
| `UNIQUEIDENTIFIER` | `StringType()` | `string` | |
| `VARBINARY` | `BinaryType()` | `binary` | |
| JSON in `NVARCHAR` | `StringType()`, a struct, or 🆕 `VARIANT` | | See §10 |

---

## 5. Filtering

**T-SQL**
```sql
SELECT * FROM orders
WHERE status = 'paid'
  AND (amount > 100 OR customer_id IN (1, 2, 3))
  AND NOT is_test = 1;
```
**PySpark**
```python
orders.filter(
    (F.col("status") == "paid")
    & ((F.col("amount") > 100) | F.col("customer_id").isin(1, 2, 3))
    & ~(F.col("is_test") == 1)
)

# Same thing as a SQL string (handy for quick filters)
orders.filter("status = 'paid' AND (amount > 100 OR customer_id IN (1, 2, 3)) AND NOT is_test = 1")
```

**Why the parentheses:** in Python, `&` `|` `~` bind *tighter* than `==` `>` `<`.
`F.col("a") > 1 & F.col("b") < 5` is parsed as `F.col("a") > (1 & F.col("b")) < 5` → error. Wrap every comparison.
⚠️ Python `and` / `or` / `not` don't work on columns → `ValueError: Cannot convert column into bool`.

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `WHERE a = 1` | `df.filter(F.col("a") == 1)` or `df.where(...)` | `filter` and `where` are identical |
| `a <> 1` | `F.col("a") != 1` | Excludes NULLs (three-valued logic, same as T-SQL) |
| `a IN (1, 2)` | `F.col("a").isin(1, 2)` or `.isin([1, 2])` | |
| `a NOT IN (1, 2)` | `~F.col("a").isin(1, 2)` | Rows where `a` is NULL are dropped too |
| `a LIKE 'AB%'` | `F.col("a").like("AB%")` | **Case-sensitive** (unlike a CI collation); use `.ilike()` 🆕 3.3+ |
| `a LIKE '%[0-9]'` | `F.col("a").rlike(r"\d$")` | Java regex; matches anywhere unless anchored with `^` `$` |
| `LIKE 'AB%'` / `'%AB'` / `'%AB%'` | `.startswith("AB")` / `.endswith("AB")` / `.contains("AB")` | |
| `a BETWEEN 1 AND 10` | `F.col("a").between(1, 10)` | Inclusive both ends; on timestamps, `between("2026-01-01", "2026-01-31")` misses Jan 31 after 00:00 |
| `a IS NULL` | `F.col("a").isNull()` | Never `== None` (always NULL → filters out everything) |
| `a IS NOT NULL` | `F.col("a").isNotNull()` | |
| `d >= '2026-01-01' AND d < '2026-02-01'` | `(F.col("d") >= "2026-01-01") & (F.col("d") < "2026-02-01")` | Strings are auto-cast to date/timestamp |
| `a IN (SELECT id FROM t2)` | `left_semi` join (§8) | Don't `collect()` a big table into a Python list for `isin` |

**Spark SQL** extras: `a <=> b` (NULL-safe equals, like T-SQL 2022's `IS NOT DISTINCT FROM`), `a ILIKE 'ab%'`.

---

## 6. Functions by category

### Strings

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `UPPER(s)` / `LOWER(s)` | `F.upper("s")` / `F.lower("s")` | |
| `LTRIM(RTRIM(s))` | `F.trim("s")`, `F.ltrim`, `F.rtrim` | Spaces only; other chars: `F.regexp_replace` |
| `LEN(s)` | `F.length("s")` | `LEN` ignores trailing spaces; `length` counts them |
| `SUBSTRING(s, 1, 3)` | `F.substring("s", 1, 3)` | 1-based like T-SQL; pos/len must be ints (for column-driven lengths: `F.col("s").substr(F.lit(1), F.col("n"))`) |
| `LEFT(s, 3)` / `RIGHT(s, 3)` | `F.substring("s", 1, 3)` / `F.substring("s", -3, 3)` | `F.left`/`F.right` 🆕 3.5 need `F.lit(3)`, a plain `3` won't do |
| `CHARINDEX('x', s)` | `F.instr("s", "x")` or `F.locate("x", "s")` | Argument order differs between the two; 0 if not found |
| `REPLACE(s, 'a', 'b')` | `F.regexp_replace("s", "a", "b")` | Pattern is regex: escape `.` `(` `$` etc. (`r"\."`) |
| `CONCAT(a, '-', b)` | `F.concat("a", F.lit("-"), "b")` | ⚠️ Returns NULL if **any** input is NULL (T-SQL `CONCAT` treats NULL as `''`) |
| `CONCAT_WS('-', a, b)` (2017+) | `F.concat_ws("-", "a", "b")` | Skips NULLs, safest choice |
| `RIGHT('00000' + s, 5)` | `F.lpad("s", 5, "0")` / `F.rpad` | Truncates values longer than 5 |
| `STRING_SPLIT(s, ',')` | `F.split("s", ",")` → array; then `F.explode` for rows | Pattern is regex: split on a dot with `r"\."` |
| `PATINDEX` / extract pattern | `F.regexp_extract("s", r"(\d+)", 1)` | Returns `''`, not NULL, when nothing matches |
| Proper case | `F.initcap("s")` | |
| `FORMAT(n, 'N2')` | `F.format_number("n", 2)` | Returns a string with thousands separators |
| `REVERSE`, `TRANSLATE` | `F.reverse`, `F.translate("s", "abc", "xyz")` | |
| `NEWID()` | `F.expr("uuid()")` | Non-deterministic: changes on every recompute |
| `HASHBYTES('SHA2_256', s)` | `F.sha2("s", 256)` / `F.md5("s")` | Returns hex string; for multi-column keys hash `F.concat_ws("\|\|", ...)` |

### Dates and timestamps

Spark format patterns: `yyyy-MM-dd HH:mm:ss`. ⚠️ `MM` = month, `mm` = minute; `dd` = day, `DD` = day-of-year.

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `GETDATE()` / `SYSDATETIME()` | `F.current_timestamp()` | Session time zone (UTC by default on Databricks) |
| `CAST(GETDATE() AS DATE)` | `F.current_date()` | |
| `CONVERT(DATE, s, 103)` | `F.to_date("s", "dd/MM/yyyy")` | Unparseable → NULL, or error under ANSI |
| `TRY_CONVERT(DATETIME2, s)` | `F.try_to_timestamp("s", F.lit("dd/MM/yyyy HH:mm"))` | 🆕 3.5+; the format must be wrapped in `F.lit` |
| `CONVERT(DATETIME2, s)` | `F.to_timestamp("s", "yyyy-MM-dd HH:mm:ss")` | |
| `DATEADD(day, 7, d)` | `F.date_add("d", 7)` / `F.date_sub("d", 7)` | Returns DATE even for timestamp input |
| `DATEADD(month, 1, d)` | `F.add_months("d", 1)` | |
| `DATEADD(hour, 2, ts)` | `F.col("ts") + F.expr("INTERVAL 2 HOURS")` | Spark SQL: `timestampadd(HOUR, 2, ts)` |
| `DATEDIFF(day, start, end)` | `F.datediff("end", "start")` | ⚠️ **Argument order reversed** vs T-SQL |
| `DATEDIFF(minute, a, b)` | `F.expr("datediff(MINUTE, a, b)")` | Spark SQL has the T-SQL-style 3-arg form (also `timestampdiff`) |
| `DATEDIFF(month, a, b)` | `F.months_between("b", "a")` | Fractional result; T-SQL counts month boundaries |
| `YEAR(d)`, `MONTH(d)`, `DAY(d)` | `F.year`, `F.month`, `F.dayofmonth` | |
| `DATEPART(hour, ts)` | `F.hour`, `F.minute`, `F.second`, `F.quarter`, `F.weekofyear` | |
| `DATEPART(weekday, d)` | `F.dayofweek("d")` | Always 1 = Sunday (no `DATEFIRST`); `F.weekday` 🆕 3.5: 0 = Monday |
| `DATENAME(month, d)` | `F.date_format("d", "MMMM")` | `"EEEE"` for day name |
| `FORMAT(d, 'yyyy-MM')` | `F.date_format("d", "yyyy-MM")` | Returns a string |
| `EOMONTH(d)` | `F.last_day("d")` | |
| `DATEFROMPARTS(y, m, d)` | `F.make_date("y", "m", "d")` | 🆕 3.5+ in Python; older: `F.expr("make_date(y, m, d)")` |
| First of month | `F.trunc("d", "month")` → date; `F.date_trunc("month", "ts")` → timestamp | ⚠️ The two take arguments in **opposite** order |
| Seconds since 1970 | `F.unix_timestamp("ts")` / `F.timestamp_seconds("n")` | `F.from_unixtime` returns a *string* |
| `ts AT TIME ZONE ...` | `F.from_utc_timestamp("ts", "America/New_York")` / `F.to_utc_timestamp` | Use region IDs, not fixed offsets |

Set the session time zone: `spark.conf.set("spark.sql.session.timeZone", "America/New_York")` (one of the configs allowed on serverless).

### Numbers

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `ROUND(x, 2)` | `F.round("x", 2)` | HALF_UP; `F.bround` = banker's rounding |
| `CEILING(x)` / `FLOOR(x)` | `F.ceil("x")` / `F.floor("x")` | |
| `ABS`, `POWER`, `SQRT`, `LOG10`, `EXP` | `F.abs`, `F.pow("x", 2)`, `F.sqrt`, `F.log10`, `F.exp` | `F.log("x")` is natural log; `F.log(2.0, "x")` = base 2 |
| `x % 2` | `F.col("x") % 2` | |
| `int / int` (truncates in T-SQL!) | `F.col("a") / F.col("b")` | Spark `/` always returns a fractional result; integer division: `F.expr("a div b")` |
| Divide by zero → error | `F.try_divide("a", "b")` | 🆕 3.5+; plain `/` gives NULL (non-ANSI) or error (ANSI) |
| `NULLIF(b, 0)` | `F.nullif("b", F.lit(0))` | 🆕 3.5+; older: `F.expr("nullif(b, 0)")` |
| `CAST(x AS DECIMAL(18,2))` | `F.col("x").cast("decimal(18,2)")` | Overflow → NULL or error (ANSI) |
| `ISNUMERIC(x)` | `F.expr("try_cast(x AS DOUBLE)").isNotNull()` | |
| `GREATEST` / `LEAST` (2022+) | `F.greatest("a", "b")` / `F.least` | Skip NULLs |
| `RAND()` | `F.rand(seed=42)` | |

### Conditional logic: `when` / `otherwise`

**T-SQL**
```sql
CASE WHEN amount >= 1000 THEN 'high'
     WHEN amount >= 100  THEN 'mid'
     ELSE 'low' END AS tier,
IIF(is_vip = 1, 'Y', 'N') AS vip_flag
```
**PySpark**
```python
df.select(
    F.when(F.col("amount") >= 1000, "high")
     .when(F.col("amount") >= 100, "mid")
     .otherwise("low").alias("tier"),
    F.when(F.col("is_vip") == 1, "Y").otherwise("N").alias("vip_flag"),
)
# Or just write the CASE in SQL:
df.withColumn("tier", F.expr("CASE WHEN amount >= 1000 THEN 'high' WHEN amount >= 100 THEN 'mid' ELSE 'low' END"))
```
⚠️ No `.otherwise()` → NULL (same as CASE without ELSE). Chain `.when()` on the previous `when`, don't call `F.when` again. All branches should return the same type.

### Null handling

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `COALESCE(a, b, 0)` | `F.coalesce("a", "b", F.lit(0))` | Strings are column names: wrap literals in `F.lit` |
| `ISNULL(a, 0)` | `F.ifnull("a", F.lit(0))` / `F.nvl` | 🆕 3.5+; `coalesce` works everywhere |
| `UPDATE t SET a = 0 WHERE a IS NULL` | `df.fillna(0, subset=["a"])` | Only fills columns whose type matches the value |
| Several defaults | `df.fillna({"qty": 0, "city": "unknown"})` | Same as `df.na.fill(...)` |
| `DELETE FROM t WHERE a IS NULL` | `df.dropna(subset=["a"])` | Default `how="any"` across **all** columns; `how="all"`, `thresh=2` also exist |
| `a = b` treating NULLs as equal | `F.col("a").eqNullSafe(F.col("b"))` | Spark SQL: `a <=> b` |

### `lit`: constants

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `SELECT 'crm' AS src` | `F.lit("crm").alias("src")` | |
| `NULL AS c` | `F.lit(None).cast("string").alias("c")` | Untyped NULL becomes VOID type; cast it or Delta writes fail |
| Constant date | `F.lit("2026-01-01").cast("date")` | |
| Python variable in an expression | `F.col("amount") > F.lit(threshold)` | Comparisons auto-wrap, but `F.lit` makes intent explicit |

---

## 7. Aggregations

**T-SQL**
```sql
SELECT customer_id,
       COUNT(*)        AS n_orders,
       SUM(amount)     AS total,
       AVG(amount)     AS avg_amount,
       MAX(order_date) AS last_order
FROM orders
GROUP BY customer_id
HAVING SUM(amount) > 1000
ORDER BY total DESC;
```
**PySpark**
```python
(orders
 .groupBy("customer_id")
 .agg(F.count("*").alias("n_orders"),
      F.sum("amount").alias("total"),
      F.avg("amount").alias("avg_amount"),
      F.max("order_date").alias("last_order"))
 .filter(F.col("total") > 1000)          # HAVING
 .orderBy(F.desc("total")))
```
⚠️ Without `.alias()` columns are named `sum(amount)` and need backticks later. Avoid the dict form `agg({"amount": "sum"})`: one aggregate per column, no aliases.

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `COUNT(*)` | `F.count("*")` | Counts rows |
| `COUNT(col)` | `F.count("col")` | Ignores NULLs |
| `COUNT(DISTINCT col)` | `F.countDistinct("col")` / `F.count_distinct` | Exact = expensive on big data |
| Approximate distinct | `F.approx_count_distinct("col")` | Much faster, ~5% error by default |
| Whole table | `SELECT SUM(x) FROM t` | `df.agg(F.sum("x"))` (no `groupBy`) |
| `SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END)` | `F.sum(F.when(F.col("status") == "paid", F.col("amount")).otherwise(0))` | |
| `COUNT(CASE WHEN ... THEN 1 END)` | `F.count_if(F.col("status") == "paid")` | 🆕 3.5+; older: `F.count(F.when(cond, 1))` |
| Value at max of another column | `F.max_by("product", "amount")` / `F.min_by` | 🆕 3.3+; replaces a ROW_NUMBER subquery |
| `STRING_AGG(x, ', ')` (2017+; on 2016 `FOR XML PATH`) | `F.array_join(F.collect_list("x"), ", ")` | Order not guaranteed: wrap in `F.array_sort`; 🆕 Spark 4 / recent DBR SQL: `listagg(x, ', ') WITHIN GROUP (ORDER BY x)` |
| Distinct list | `F.collect_set("x")` | Unordered |
| `PERCENTILE_CONT(0.5)` | `F.percentile_approx("x", 0.5)` / `F.median("x")` 🆕 3.4 | |
| `ROLLUP` / `CUBE` | `df.rollup("a", "b").agg(...)` / `df.cube(...)` | Subtotal rows have NULL in the rolled-up columns |

### Pivot / unpivot

**T-SQL**
```sql
SELECT * FROM (SELECT customer_id, month, amount FROM sales) s
PIVOT (SUM(amount) FOR month IN ([Jan], [Feb], [Mar])) p;
```
**PySpark**
```python
sales.groupBy("customer_id").pivot("month", ["Jan", "Feb", "Mar"]).agg(F.sum("amount"))

# Unpivot 🆕 3.4+
wide.unpivot(ids=["customer_id"], values=["Jan", "Feb", "Mar"],
             variableColumnName="month", valueColumnName="amount")
```
**Spark SQL**
```sql
SELECT * FROM (SELECT customer_id, month, amount FROM sales)
PIVOT (SUM(amount) FOR month IN ('Jan', 'Feb', 'Mar'));
```
⚠️ Always pass the value list to `pivot()`; otherwise Spark runs an extra job to find distinct values. In Spark SQL, every non-pivoted column becomes a grouping column, so select only what you need in a subquery first (same as T-SQL).

---

## 8. Joins

| T-SQL | PySpark `how=` | ⚠️ Gotcha |
|---|---|---|
| `INNER JOIN` | `"inner"` (default) | |
| `LEFT JOIN` | `"left"` | |
| `RIGHT JOIN` | `"right"` | |
| `FULL OUTER JOIN` | `"full"` | |
| `WHERE EXISTS (...)` | `"left_semi"` (exam) | Returns left columns only, never duplicates left rows |
| `WHERE NOT EXISTS (...)` | `"left_anti"` (exam) | NULL-safe, unlike `NOT IN` |
| `CROSS JOIN` | `a.crossJoin(b)` | Rows multiply: n × m |
| `CROSS APPLY OPENJSON / STRING_SPLIT` | `F.explode` (§10) | |
| `OUTER APPLY` | `F.explode_outer` or a left join | |

### Same key name vs different key names

```python
# Same name: pass a string or list → one key column in the output (like USING)
orders.join(customers, on="customer_id", how="left")
orders.join(customers, on=["customer_id", "region"], how="inner")

# Different names: alias both sides and write the condition
o, c = orders.alias("o"), customers.alias("c")
(o.join(c, F.col("o.cust_id") == F.col("c.id"), "left")
  .select("o.*", F.col("c.name").alias("customer_name")))

# Composite / non-equi condition (e.g. slowly changing dimension lookup)
o.join(c, (F.col("o.cust_id") == F.col("c.id"))
          & F.col("o.order_date").between(F.col("c.valid_from"), F.col("c.valid_to")), "left")
```

### EXISTS / NOT EXISTS → semi / anti

**T-SQL**
```sql
SELECT * FROM customers c WHERE EXISTS     (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
SELECT * FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
```
**PySpark**
```python
customers.join(orders, "customer_id", "left_semi")   # customers with at least one order
customers.join(orders, "customer_id", "left_anti")   # customers with no orders
```
**Spark SQL**: `SELECT * FROM customers c LEFT ANTI JOIN orders o ON o.customer_id = c.customer_id` (`EXISTS` subqueries also work).

### Avoiding ambiguous columns

```python
# ❌ Both DataFrames have customer_id → AMBIGUOUS_REFERENCE when you select it
bad = orders.join(customers, orders.customer_id == customers.customer_id)

# ✅ Fix 1: join on the name (one key column kept)
orders.join(customers, "customer_id")
# ✅ Fix 2: aliases + qualified names
o.join(c, F.col("o.customer_id") == F.col("c.customer_id")).select("o.*", "c.name")
# ✅ Fix 3: drop the duplicate using a DataFrame-bound reference
bad.drop(customers["customer_id"])
```

### Broadcast joins

```python
orders.join(F.broadcast(dim_country), "country_code")   # ships the small table to every worker
```
Spark SQL hint: `SELECT /*+ BROADCAST(d) */ * FROM orders o JOIN dim_country d ON ...`
Spark broadcasts automatically below ~10 MB, and AQE can switch to a broadcast join at runtime.
⚠️ Only broadcast genuinely small tables (MBs, not GBs). ⚠️ NULL keys never match (same as T-SQL); duplicate keys on the right multiply rows on the left.

---

## 9. Window functions

```python
w = Window.partitionBy("customer_id").orderBy(F.col("order_ts").desc())
```

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `ROW_NUMBER() OVER (PARTITION BY c ORDER BY ts DESC)` | `F.row_number().over(w)` | Ties are broken arbitrarily: add a tiebreaker column |
| `RANK() OVER (...)` | `F.rank().over(w)` | Gaps after ties (1, 1, 3) |
| `DENSE_RANK() OVER (...)` | `F.dense_rank().over(w)` | No gaps (1, 1, 2) |
| `NTILE(4) OVER (...)` | `F.ntile(4).over(w)` | |
| `LAG(amount, 1, 0) OVER (...)` | `F.lag("amount", 1, 0).over(w)` | 3rd arg is the default when there's no previous row |
| `LEAD(amount) OVER (...)` | `F.lead("amount").over(w)` | |
| `SUM(amount) OVER (PARTITION BY c)` | `F.sum("amount").over(Window.partitionBy("c"))` | No `orderBy` = whole partition |
| `FIRST_VALUE(x)` | `F.first("x", ignorenulls=True).over(w)` | |
| `LAST_VALUE(x)` | `F.last("x").over(w_full)` | Needs a full frame (below), as in T-SQL |

### Running totals and frames

**T-SQL**
```sql
SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_ts
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
```
**PySpark**
```python
w_run  = (Window.partitionBy("customer_id").orderBy("order_ts")
          .rowsBetween(Window.unboundedPreceding, Window.currentRow))
w_3row = Window.partitionBy("customer_id").orderBy("order_ts").rowsBetween(-2, 0)   # 3-row moving window
w_full = (Window.partitionBy("customer_id").orderBy("order_ts")
          .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))

df = (df
      .withColumn("running_total", F.sum("amount").over(w_run))
      .withColumn("moving_avg_3",  F.avg("amount").over(w_3row))
      .withColumn("pct_of_customer", F.col("amount") / F.sum("amount").over(Window.partitionBy("customer_id"))))

# Value-based range: last 7 days (orderBy a numeric version of the timestamp)
w_7d = (Window.partitionBy("customer_id").orderBy(F.col("order_ts").cast("long"))
        .rangeBetween(-7 * 86400, 0))
```
⚠️ With `orderBy` and no explicit frame, the default is `RANGE UNBOUNDED PRECEDING` (same as T-SQL), so rows with tied `order_ts` are summed together. Use `rowsBetween` for true running totals.
⚠️ A window with no `partitionBy` pulls all data into one partition (Spark logs a warning).

### Dedup: keep the latest record per key (exam pattern)

**T-SQL**
```sql
WITH x AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY updated_at DESC) AS rn
  FROM staging_orders)
SELECT * FROM x WHERE rn = 1;
```
**PySpark**
```python
w = Window.partitionBy("order_id").orderBy(F.col("updated_at").desc(), F.col("ingested_at").desc())
latest = (staging_orders
          .withColumn("rn", F.row_number().over(w))
          .filter(F.col("rn") == 1)
          .drop("rn"))
```
**Spark SQL** (no CTE needed thanks to `QUALIFY`):
```sql
SELECT * FROM staging_orders
QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY updated_at DESC) = 1;
```

---

## 10. Complex and nested data

Sample record used below:
```json
{"order_id": 1,
 "customer": {"id": 7, "name": "Alice", "address": {"city": "Springfield"}},
 "items": [{"sku": "A1", "qty": 2, "price": 10.5}, {"sku": "B2", "qty": 1, "price": 99.0}],
 "tags": {"channel": "web", "promo": "NOV"}}
```
Types: `customer` = STRUCT, `items` = ARRAY<STRUCT>, `tags` = MAP<STRING,STRING>.

### Structs (dot access)

| Task | PySpark | ⚠️ Gotcha |
|---|---|---|
| One field | `F.col("customer.name")` or `F.col("customer")["name"]` | Output column is named `name`, alias it |
| Nested field | `F.col("customer.address.city").alias("city")` | |
| All fields as columns | `df.select("order_id", "customer.*")` | Name clashes if two structs share field names |
| Build a struct | `F.struct("a", "b").alias("s")` | |
| Change one field | `F.col("customer").withField("name", F.upper("customer.name"))` | 🆕 3.1+ |

### Arrays

| Task | PySpark | ⚠️ Gotcha |
|---|---|---|
| By index | `F.col("items")[0]` | **0-based**; out of range → NULL (non-ANSI) |
| By position | `F.element_at("items", 1)` | **1-based**; `-1` = last; out of range errors under ANSI. 🆕 `F.get("items", 0)` (3.4+) is 0-based and NULL-safe |
| Length | `F.size("items")` | NULL array → `-1` on legacy settings, NULL under ANSI |
| Contains | `F.array_contains("skus", "A1")` | |
| Set ops | `F.array_distinct`, `F.array_union`, `F.array_except`, `F.arrays_overlap` | |
| Sort / slice / flatten | `F.array_sort`, `F.slice("a", 1, 3)`, `F.flatten` | `slice` is 1-based |
| Map each element | `F.transform("items", lambda x: x["qty"] * x["price"])` | Higher-order functions run in Spark, not Python: fast |
| Filter elements | `F.filter("items", lambda x: x["qty"] > 1)` | |
| Reduce | `F.aggregate("items", F.lit(0.0), lambda acc, x: acc + x["qty"] * x["price"])` | Initial value type must match the result type, or cast |
| Any element matches | `F.exists("items", lambda x: x["sku"] == "A1")` | |

### Maps

| Task | PySpark |
|---|---|
| Value by key | `F.col("tags")["channel"]` or `F.col("tags").getItem("channel")` |
| Keys / values | `F.map_keys("tags")`, `F.map_values("tags")` |
| Build | `F.create_map(F.lit("channel"), F.col("channel"))` |
| To rows | `df.select("order_id", F.explode("tags"))` → columns `key`, `value` |

### explode, explode_outer, posexplode

**T-SQL**
```sql
SELECT o.order_id, i.sku, i.qty, i.price
FROM orders o
CROSS APPLY OPENJSON(o.items) WITH (sku NVARCHAR(20) '$.sku', qty INT '$.qty', price DECIMAL(10,2) '$.price') i;
```
**PySpark**
```python
(orders
 .select("order_id", F.explode("items").alias("item"))     # one row per array element
 .select("order_id", "item.sku", "item.qty", "item.price"))

orders.select("order_id", F.inline("items"))                # 🆕 3.4+: array<struct> straight to columns
orders.select("order_id", F.explode_outer("items").alias("item"))     # keeps orders with NULL/empty arrays (OUTER APPLY)
orders.select("order_id", F.posexplode("items").alias("pos", "item")) # pos is 0-based
```
**Spark SQL**
```sql
SELECT order_id, item.sku, item.qty FROM (SELECT order_id, explode(items) AS item FROM orders);
SELECT order_id, i.* FROM orders LATERAL VIEW explode(items) t AS i;   -- Hive-style, also valid
```
⚠️ `explode` drops rows whose array is NULL or empty. ⚠️ Exploding two arrays in the same query produces a cartesian product of their elements.

### JSON strings: from_json, to_json, schema_of_json, get_json_object

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `JSON_VALUE(raw, '$.customer.name')` | `F.get_json_object("raw", "$.customer.name")` | Always returns a string |
| `OPENJSON(raw) WITH (...)` | `F.from_json("raw", ddl_schema)` → struct | Malformed JSON → NULL, no error |
| `FOR JSON PATH` | `F.to_json(F.struct("order_id", "customer"))` | |
| `ISJSON(raw)` | `F.from_json("raw", ddl_schema).isNotNull()` | |
| `OPENJSON(raw)` (no WITH: key/value) | `F.explode(F.from_json("raw", "MAP<STRING,STRING>"))` | |

```python
ddl = """order_id INT,
         customer STRUCT<id: INT, name: STRING, address: STRUCT<city: STRING>>,
         items ARRAY<STRUCT<sku: STRING, qty: INT, price: DOUBLE>>,
         tags MAP<STRING, STRING>"""

parsed = raw_df.withColumn("j", F.from_json("raw", ddl)).select("j.*")

# Don't know the schema? Infer it from one sample string (a literal, not a column)
sample = raw_df.select("raw").first()[0]
parsed = raw_df.withColumn("j", F.from_json("raw", F.schema_of_json(F.lit(sample))))
print(spark.range(1).select(F.schema_of_json(F.lit(sample))).first()[0])   # copy it as a DDL schema
```

**Databricks SQL shortcuts** (exam): query JSON strings directly with `:` paths.
```sql
SELECT raw:customer.name, raw:items[0].sku, raw:items[0].qty::int FROM bronze_events;
```
🆕 **VARIANT** type (recent DBR): semi-structured data stored efficiently, no schema needed up front.
```sql
SELECT parse_json(raw) AS v FROM bronze_events;                         -- store as VARIANT
SELECT variant_get(v, '$.customer.name', 'string') FROM silver_events;  -- or v:customer.name::string
```

### Flattening nested JSON into columns

```python
def flatten(df, sep="_"):
    """Expand structs into prefixed columns and explode arrays, until nothing is nested.
    ⚠️ Each extra array multiplies rows; MapType columns are left as-is."""
    while True:
        nested = [(f.name, f.dataType) for f in df.schema.fields
                  if isinstance(f.dataType, (StructType, ArrayType))]
        if not nested:
            return df
        name, dtype = nested[0]
        if isinstance(dtype, StructType):
            cols = []
            for c in df.columns:
                if c == name:
                    cols += [F.col(f"`{name}`.`{s.name}`").alias(f"{name}{sep}{s.name}")
                             for s in dtype.fields]
                else:
                    cols.append(F.col(f"`{c}`"))
            df = df.select(cols)
        else:  # ArrayType
            df = df.withColumn(name, F.explode_outer(F.col(f"`{name}`")))

flat = flatten(parsed)
# order_id | customer_id | customer_name | customer_address_city | items_sku | items_qty | items_price | tags
```

---

## 11. Set operations and dedup

| T-SQL | PySpark | ⚠️ Gotcha |
|---|---|---|
| `UNION ALL` | `df1.union(df2)` | ⚠️ Matches columns **by position**, not name (same as T-SQL, but easy to forget) |
| `UNION ALL` by column name | `df1.unionByName(df2)` | Safer default |
| Different column sets | `df1.unionByName(df2, allowMissingColumns=True)` | Missing columns filled with NULL |
| `UNION` | `df1.union(df2).distinct()` | `union` never removes duplicates |
| `SELECT DISTINCT *` | `df.distinct()` | |
| Distinct on some columns | `df.dropDuplicates(["customer_id", "email"])` | ⚠️ Which row survives is arbitrary; use `row_number` (§9) to keep the latest |
| `INTERSECT` | `df1.intersect(df2)` | Distinct; `intersectAll` keeps duplicates |
| `EXCEPT` | `df1.subtract(df2)` | = `EXCEPT DISTINCT`; `exceptAll` keeps duplicate counts |

**Spark SQL:** `UNION ALL`, `UNION`, `INTERSECT`, `EXCEPT` (or `MINUS`) work as in T-SQL.

```python
from functools import reduce
from pyspark.sql import DataFrame
all_months = reduce(DataFrame.unionByName, [jan_df, feb_df, mar_df])   # union a list of DataFrames
```

---

## 12. Writing data

```python
df.write.mode("overwrite").saveAsTable("main.silver.orders")   # Delta is the default format on Databricks
```

| Mode | Behaviour | T-SQL equivalent |
|---|---|---|
| `"append"` | Add rows | `INSERT INTO t SELECT ...` |
| `"overwrite"` | Replace all rows (Delta: atomic, old version kept in history) | `TRUNCATE TABLE t; INSERT INTO t SELECT ...` |
| `"error"` / `"errorifexists"` (default) | Fail if the table exists | `SELECT ... INTO t` |
| `"ignore"` | Do nothing if the table exists | `IF OBJECT_ID('t') IS NULL SELECT ... INTO t` |

### `saveAsTable` vs saving to a path

| | `saveAsTable("cat.sch.t")` | `.save(path)` / `.csv(path)` / `.parquet(path)` |
|---|---|---|
| Result | Registered Unity Catalog table, governed | Just files in a folder |
| Use for | Bronze/silver/gold tables | Exports, hand-offs to other systems |

```python
# Export files to a volume (writes a FOLDER of part-*.csv files)
(df.coalesce(1).write.mode("overwrite")
   .option("header", True)
   .csv("/Volumes/main/exports/files/orders_csv/"))

# Exactly one named file, small data only: volumes are mounted as a local path
df.limit(100_000).toPandas().to_csv("/Volumes/main/exports/files/orders.csv", index=False)

# DataFrameWriterV2 alternative
df.writeTo("main.silver.orders").createOrReplace()   # .append(), .overwrite(condition) also exist
```
⚠️ `.coalesce(1)` still produces a folder, not a single named file.
⚠️ `df.write.insertInto("t")` matches columns **by position**; prefer `saveAsTable` with `mode("append")`.

### `partitionBy` and clustering

```python
df.write.mode("overwrite").partitionBy("order_date").saveAsTable("main.silver.orders")
```
⚠️ Partitioning only pays off on very large tables (Databricks guidance: roughly 1 TB+, with each partition ≥ 1 GB). A high-cardinality partition column creates millions of tiny files.
🆕 Prefer **liquid clustering** for new tables (exam):
```sql
CREATE TABLE main.silver.orders (order_id BIGINT, customer_id BIGINT, order_date DATE, amount DECIMAL(18,2))
CLUSTER BY (customer_id, order_date);
ALTER TABLE main.silver.orders CLUSTER BY (customer_id);   -- change keys without rewriting
-- 🆕 CLUSTER BY AUTO lets Databricks choose keys; Spark 4 also adds df.write.clusterBy(...)
```

### `overwriteSchema` vs `mergeSchema`

| Option | Use with | Effect |
|---|---|---|
| `.option("mergeSchema", True)` | `append` (or overwrite) | Adds new columns; existing columns stay |
| `.option("overwriteSchema", True)` | `overwrite` | Replaces the schema entirely: change types, drop or rename columns |
| neither | | Schema mismatch → `AnalysisException` (Delta schema enforcement, exam) |

```python
df.write.mode("append").option("mergeSchema", True).saveAsTable("main.silver.orders")
df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable("main.silver.orders")

# Overwrite only a slice, atomically (T-SQL: BEGIN TRAN; DELETE ... WHERE; INSERT ...; COMMIT)
(df.write.mode("overwrite")
   .option("replaceWhere", "order_date >= '2026-11-01' AND order_date < '2026-12-01'")
   .saveAsTable("main.silver.orders"))
```

**Spark SQL equivalents:** `INSERT INTO t SELECT ...`, `INSERT OVERWRITE t SELECT ...`, `CREATE OR REPLACE TABLE t AS SELECT ...`. 🆕 `INSERT INTO t BY NAME SELECT ...` matches columns by name.

---

## 13. Delta Lake essentials (exam)

### MERGE INTO (upsert)

**T-SQL**
```sql
MERGE dbo.customers AS t
USING #updates AS s
  ON t.customer_id = s.customer_id
WHEN MATCHED AND s.is_deleted = 1 THEN DELETE
WHEN MATCHED THEN UPDATE SET t.email = s.email, t.city = s.city
WHEN NOT MATCHED BY TARGET THEN INSERT (customer_id, email, city) VALUES (s.customer_id, s.email, s.city)
WHEN NOT MATCHED BY SOURCE THEN DELETE;
```
**Spark SQL**
```sql
-- first: updates_df.createOrReplaceTempView("updates")
MERGE INTO main.silver.customers AS t
USING updates AS s
  ON t.customer_id = s.customer_id
WHEN MATCHED AND s.is_deleted = true THEN DELETE
WHEN MATCHED THEN UPDATE SET *              -- all columns by name
WHEN NOT MATCHED THEN INSERT *
WHEN NOT MATCHED BY SOURCE THEN DELETE;     -- optional
```
**Python DeltaTable API**
```python
from delta.tables import DeltaTable

tgt = DeltaTable.forName(spark, "main.silver.customers")
(tgt.alias("t")
    .merge(updates_df.alias("s"), "t.customer_id = s.customer_id")
    .whenMatchedDelete(condition="s.is_deleted = true")
    .whenMatchedUpdateAll()                      # or .whenMatchedUpdate(set={"email": "s.email", "updated_at": "current_timestamp()"})
    .whenNotMatchedInsertAll()                   # or .whenNotMatchedInsert(values={...})
    .whenNotMatchedBySourceDelete()              # optional
    .execute())
```
⚠️ If **several source rows match one target row**, MERGE fails (`DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE`), same as T-SQL. Dedupe the source with `row_number` first (§9).
⚠️ Clauses are evaluated in order; the first matching `WHEN` wins. `UPDATE SET *` / `INSERT *` require matching column names.
🆕 Schema evolution in MERGE: `MERGE WITH SCHEMA EVOLUTION INTO ...` (SQL) or `.withSchemaEvolution()` (Python) on recent runtimes.
⚠️ T-SQL `UPDATE t SET ... FROM t JOIN s ...` has no direct Delta equivalent: use `MERGE`. Plain `UPDATE t SET ... WHERE` and `DELETE FROM t WHERE` work as in T-SQL.

### History, time travel, restore

| Task | Spark SQL | Python / T-SQL analogy |
|---|---|---|
| Version history | `DESCRIBE HISTORY main.silver.customers` | `DeltaTable.forName(spark, t).history(10)` |
| File-level details | `DESCRIBE DETAIL main.silver.customers` | location, numFiles, sizeInBytes |
| Read old version | `SELECT * FROM t VERSION AS OF 5` (or `t@v5`) | `spark.read.option("versionAsOf", 5).table(t)` |
| Read as of time | `SELECT * FROM t TIMESTAMP AS OF '2026-11-01'` | T-SQL temporal tables: `FOR SYSTEM_TIME AS OF` |
| Roll back | `RESTORE TABLE t TO VERSION AS OF 5` | Restore is itself a new version, so it can be undone |

### OPTIMIZE and VACUUM

```sql
OPTIMIZE main.silver.orders;                          -- compact small files (≈ ALTER INDEX ... REBUILD)
OPTIMIZE main.silver.orders ZORDER BY (customer_id);  -- co-locate data; not used with liquid clustering
VACUUM main.silver.orders;                            -- delete unreferenced files older than 7 days (default)
VACUUM main.silver.orders RETAIN 240 HOURS;
VACUUM main.silver.orders DRY RUN;                    -- list what would be deleted
```
⚠️ After VACUUM you **can't time travel** to versions whose files were removed. Retention below 7 days is blocked by a safety check.
🆕 **Predictive optimization** runs OPTIMIZE / VACUUM / ANALYZE automatically on Unity Catalog managed tables when enabled (default for newer accounts).

### Managed vs external tables

| | Managed | External |
|---|---|---|
| Create | `CREATE TABLE main.silver.t (...)` | `CREATE TABLE main.silver.t (...) LOCATION 'abfss://...'` (needs an external location) |
| `DROP TABLE` | Data deleted too (UC: `UNDROP TABLE` within 7 days 🆕) | Only metadata removed; files stay |
| Who manages files | Unity Catalog | You |
| Recommendation | Default choice | When other tools must own or read the files directly |

### CTAS and related DDL

| T-SQL | Spark SQL | Note |
|---|---|---|
| `SELECT ... INTO new_t FROM src` | `CREATE OR REPLACE TABLE main.silver.new_t AS SELECT ... FROM src` | CTAS: schema comes from the query (exam) |
| `DROP` + `CREATE` | `CREATE OR REPLACE TABLE ...` | Keeps table history and permissions; DROP+CREATE loses them (exam) |
| `TRUNCATE` + `INSERT` | `INSERT OVERWRITE t SELECT ...` | Fails if the schema differs; CREATE OR REPLACE doesn't (exam) |
| Copy a table | `CREATE TABLE t_dev DEEP CLONE t` | Copies data and metadata (exam) |
| Zero-copy test copy | `CREATE TABLE t_dev SHALLOW CLONE t` | Metadata only; points at source files (exam) |
| `IDENTITY(1,1)` | `id BIGINT GENERATED ALWAYS AS IDENTITY` | Unique but not gap-free or consecutive |
| `CHECK` constraint | `ALTER TABLE t ADD CONSTRAINT pos_amt CHECK (amount >= 0)` | Enforced: violating writes fail |
| `PRIMARY KEY` / `FOREIGN KEY` | Same syntax on UC tables | ⚠️ **Informational only, not enforced**: duplicates are allowed |

---

## 14. Mixing SQL and Python

| Task | Code | ⚠️ Gotcha |
|---|---|---|
| Run SQL, get a DataFrame | `df = spark.sql("SELECT * FROM main.silver.orders")` | One statement per call (no T-SQL-style batches) |
| Table → DataFrame | `spark.table("main.silver.orders")` | Same as `spark.read.table` |
| DataFrame → SQL name | `df.createOrReplaceTempView("orders_v")` | Session-scoped *view*: recomputed each query, not materialized like `#temp` |
| Whole cell in SQL | `%sql` on the first line of a cell | Result of the last `%sql` cell is available in Python as `_sqldf` |
| Set default namespace | `USE CATALOG main; USE SCHEMA silver;` | Python: `spark.catalog.setCurrentCatalog("main")` |
| Does a table exist? | `spark.catalog.tableExists("main.silver.orders")` | `IF OBJECT_ID(...) IS NOT NULL` |

⚠️ Global temp views (`global_temp`) aren't supported on serverless; regular temp views are.

### Dynamic SQL: f-strings vs parameters

**T-SQL**
```sql
DECLARE @sql NVARCHAR(MAX) = N'SELECT * FROM ' + QUOTENAME(@tbl) + N' WHERE amount > @min';
EXEC sp_executesql @sql, N'@min INT', @min = 100;
```
**PySpark**
```python
tbl, min_amt = "main.silver.orders", 100

# f-string: fine for trusted identifiers you control; ⚠️ injection-prone for values
df = spark.sql(f"SELECT * FROM {tbl} WHERE amount > {min_amt}")

# ✅ Parameterized (like sp_executesql): named markers + IDENTIFIER() for object names 🆕
df = spark.sql("SELECT * FROM IDENTIFIER(:tbl) WHERE amount > :min_amt",
               args={"tbl": tbl, "min_amt": min_amt})

# Reference a DataFrame directly inside SQL
spark.sql("SELECT status, COUNT(*) AS n FROM {src} GROUP BY status", src=orders_df)
```
⚠️ With f-strings, string values need their own quotes: `f"WHERE status = '{status}'"`. Parameters handle quoting for you.

**Pure SQL variables and dynamic SQL** 🆕 (recent DBR):
```sql
DECLARE VARIABLE min_amt INT DEFAULT 100;          -- like DECLARE @min_amt INT = 100
SET VAR min_amt = 250;
SELECT * FROM main.silver.orders WHERE amount > min_amt;

DECLARE VARIABLE tbl STRING DEFAULT 'main.silver.orders';
EXECUTE IMMEDIATE 'SELECT COUNT(*) FROM IDENTIFIER(?)' USING tbl;   -- like sp_executesql
```
Recent runtimes also support SQL scripting (`BEGIN ... END` blocks with `IF`, loops), close to a T-SQL batch.

---

## 15. Common errors and gotchas

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `SyntaxError` / keyword can't be an expression | `F.col("a") = 1` | Python comparison is `==`. (Inside SQL strings like `"a = 1"`, `=` is fine.) |
| 2 | Data type mismatch / odd error in a filter | `F.col("a") > 1 & F.col("b") < 5` | Parenthesize every comparison: `(F.col("a") > 1) & (F.col("b") < 5)` |
| 3 | `ValueError: Cannot convert column into bool` | `and` / `or` / `not` / `if col:` on Columns | Use `&` `\|` `~`; `F.when` instead of `if` |
| 4 | `AMBIGUOUS_REFERENCE` | Same column name on both sides of a join | Join on the name, alias both DataFrames, or drop one copy (§8) |
| 5 | Filter returns nothing | `F.col("a") == None` (NULL comparison is never true) | `F.col("a").isNull()` |
| 6 | Delta write fails on a VOID column | `F.lit(None)` with no type | `F.lit(None).cast("string")` |
| 7 | Driver out of memory / notebook hangs | `collect()` / `toPandas()` on a big DataFrame pulls every row into one machine | Aggregate or `limit()` first; `display()` for looking; write results to a table |
| 8 | Option seems ignored | Shortcut method's own default overrides your earlier `.option()` | See below |
| 9 | String filter misses rows | Spark string comparison is **case-sensitive** (SQL Server's default collation isn't) | `F.lower("a") == "x"`, `.ilike()`, or 🆕 collations (`COLLATE UTF8_LCASE`) on recent runtimes |
| 10 | `AnalysisException` / `UNRESOLVED_COLUMN` / `TABLE_OR_VIEW_NOT_FOUND` | Typo, wrong namespace, column dropped earlier | Like a T-SQL compile error: check `df.columns`, use full `catalog.schema.table` |
| 11 | Runtime error deep in a job | Bad cast under ANSI, divide by zero, `FAILFAST` malformed row, Python UDF exception | Only appears when an action runs; use `try_*` functions; inspect the source rows |
| 12 | `df` is `None` | `df = df.show()` (show prints and returns `None`) | Call `df.show()` on its own line |
| 13 | Column with a dot or space "not found" | `.` means struct access; spaces need quoting | Backticks: ``F.col("`order.id`")``, or rename the column |
| 14 | Loop over rows is painfully slow | `for row in df.collect():` is a cursor (RBAR) | Express it as set-based DataFrame operations |
| 15 | Notebook slows as columns are added | `withColumn` called in a long loop | One `select(...)` or `withColumns({...})` |
| 16 | `[NOT_SUPPORTED_WITH_SERVERLESS]` / `sparkContext` errors 🆕 | RDD APIs, `cache()`, unsupported `spark.conf` keys | Use DataFrame APIs; write intermediates to Delta tables |

### #8 in detail: options silently overridden

```python
spark.read.option("wholetext", True).text(path)        # ❌ still line by line
# .text(paths, wholetext=False, ...) re-sets wholetext=False after your .option() call

spark.read.text(path, wholetext=True)                  # ✅ pass it to the method
spark.read.format("text").option("wholetext", True).load(path)   # ✅ or use the generic form
```
Rule of thumb: either pass options as **keyword arguments to the shortcut method** (`.csv(path, header=True)`), or use the generic **`format().option().load()`** chain. Don't mix the two. Also, a later `.option("x", ...)` overrides an earlier one with the same key.

### None vs null

| Python | Spark |
|---|---|
| `None` in data you create | becomes SQL `NULL` |
| `df.filter(F.col("a").isNull())` | `WHERE a IS NULL` |
| `float("nan")` | `NaN` is **not** NULL: `isNull()` won't find it; use `F.isnan("a")` |

```python
spark.createDataFrame([(1, None), (2, "x")], "id INT, code STRING")   # give a schema so None has a type
```

### AnalysisException vs runtime errors

| | AnalysisException | Runtime error |
|---|---|---|
| T-SQL analogy | Compile / binding error | Error during execution |
| When | Plan is analyzed | An action executes |
| Examples | Unknown column or table, type mismatch, schema mismatch on write | Cast failure (ANSI), divide by zero, OOM, malformed records with `FAILFAST` |

🆕 On serverless (Spark Connect), even analysis is deferred: an `AnalysisException` from line 3 may only appear at the `display()` on line 20. To locate a problem, call `df.printSchema()` or `df.columns` right after a suspect step; that forces analysis without running the job.

---

## 16. Performance basics

### Partitions

A DataFrame is split into **partitions**; each is processed by one task in parallel.

| Method | Shuffle? | Can increase? | Use for |
|---|---|---|---|
| `df.repartition(200)` / `repartition("customer_id")` | Yes, full shuffle | Yes | Balancing skewed data; grouping by a key before a heavy operation |
| `df.coalesce(8)` | No | No, only decreases | Fewer output files before writing |

```python
df.select(F.spark_partition_id().alias("p")).groupBy("p").count()   # rows per partition (works on serverless)
df.rdd.getNumPartitions()                                            # classic compute only (no RDDs on serverless)
```
🆕 On serverless, AQE sizes shuffle partitions automatically and `spark.sql.shuffle.partitions` tuning is rarely needed.

### Caching

```python
# Classic compute:
df.cache(); df.count()     # cache is lazy: an action fills it
df.unpersist()

# Serverless: cache()/persist() aren't supported 🆕. Materialize instead:
df.write.mode("overwrite").saveAsTable("main.scratch.orders_stage")
stage = spark.table("main.scratch.orders_stage")
```
Only worth it when the same expensive DataFrame is reused several times.

### explain() (the execution plan)

```python
df.explain()                    # physical plan
df.explain(mode="formatted")    # readable, like SSMS's actual plan in text form
```
What to look for: `Exchange` = shuffle (expensive); `BroadcastHashJoin` (good for small dims) vs `SortMergeJoin`; `PushedFilters` / `PartitionFilters` = filters pushed down to the files. In the UI, the **query profile** shows actual time per operator.

### Avoid Python UDFs when a built-in exists

```python
# ❌ Python UDF: every row is serialized to a Python process and back
from pyspark.sql.functions import udf
clean = udf(lambda s: s.strip().upper() if s else None, StringType())
df.withColumn("code", clean("code"))

# ✅ Built-in: runs inside the engine, optimized
df.withColumn("code", F.upper(F.trim("code")))
```
If you truly need custom logic: a **pandas UDF** (`@F.pandas_udf`, vectorized), or a SQL UDF registered in Unity Catalog (`CREATE FUNCTION main.util.clean_code(s STRING) RETURNS STRING RETURN upper(trim(s))`). The parallel in SQL Server is scalar UDFs vs inline expressions.

### Quick wins

- Filter early and select only needed columns: Delta/Parquet then skip whole files and columns.
- Read tables by name so data skipping and clustering statistics are used.
- `df.isEmpty()` instead of `df.count() == 0`.
- Broadcast small dimension tables; avoid joining on skewed keys like a catch-all `'UNKNOWN'`.
- `F.approx_count_distinct` when an exact count isn't required.
- Too many small files → `OPTIMIZE` (or rely on predictive optimization).

---

## Most used 20

| # | Operation | One-liner |
|---|---|---|
| 1 | Read a table | `df = spark.table("main.silver.orders")` |
| 2 | Read files | `spark.read.format("csv").option("header", True).schema(ddl).load("/Volumes/main/raw/landing/x/")` |
| 3 | Look at it | `display(df); df.printSchema()` |
| 4 | Select + rename | `df.select("id", F.col("cust_id").alias("customer_id"))` |
| 5 | Filter | `df.filter((F.col("amount") > 0) & F.col("status").isin("paid", "shipped"))` |
| 6 | Add a column | `df.withColumn("gross", F.col("amount") * 1.14)` |
| 7 | Cast | `F.col("amount").cast("decimal(18,2)")` |
| 8 | CASE WHEN | `F.when(F.col("amount") > 100, "big").otherwise("small")` |
| 9 | Null default | `F.coalesce("city", F.lit("unknown"))` |
| 10 | Parse / format dates | `F.to_date("d", "dd/MM/yyyy")` · `F.date_format("ts", "yyyy-MM")` |
| 11 | Group + aggregate | `df.groupBy("k").agg(F.sum("x").alias("total"), F.count("*").alias("n"))` |
| 12 | Sort + top | `df.orderBy(F.desc("total")).limit(10)` |
| 13 | Join | `orders.join(customers, "customer_id", "left")` |
| 14 | EXISTS / NOT EXISTS | `a.join(b, "id", "left_semi")` · `a.join(b, "id", "left_anti")` |
| 15 | Keep latest per key | `df.withColumn("rn", F.row_number().over(Window.partitionBy("id").orderBy(F.desc("ts")))).filter("rn = 1").drop("rn")` |
| 16 | Array to rows | `df.select("id", F.explode("items").alias("item")).select("id", "item.*")` |
| 17 | Parse JSON string | `df.withColumn("j", F.from_json("raw", ddl)).select("j.*")` |
| 18 | SQL from Python | `spark.sql("SELECT * FROM IDENTIFIER(:t) WHERE d = :d", args={"t": tbl, "d": run_date})` |
| 19 | Write a table | `df.write.mode("overwrite").saveAsTable("main.silver.orders")` |
| 20 | Upsert | `DeltaTable.forName(spark, t).alias("t").merge(src.alias("s"), "t.id = s.id").whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()` |
