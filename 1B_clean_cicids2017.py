import os
from functools import reduce
import operator

from pyspark.sql import SparkSession, functions as F
from utils.schema import CICIDS2017_FULL_SCHEMA


input_dir_path = os.path.join("data", "1A_merge_cicids2017")
output_dir_path = os.path.join("data", "1B_clean_cicids2017")


spark = SparkSession.builder \
    .appName("CICIDS2017 Cleaning 1B") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


input_path = os.path.join(input_dir_path, "1A_merged_cicids2017.csv")

df = (
    spark.read
        .format("csv")
        .schema(CICIDS2017_FULL_SCHEMA)
        .option("header", True)
        .option("sep", ",")
        .load(input_path)
)

before_rows = df.count()


dot_cols = [c for c in df.columns if "." in c]
rename_map = {c: c.replace(".", "_") for c in dot_cols}
for old, new in rename_map.items():
    df = df.withColumnRenamed(old, new)


all_null = reduce(operator.and_, [F.col(c).isNull() for c in df.columns])
df = df.filter(~all_null)

df = df.filter(F.col(" Label").isNotNull() & (F.trim(F.col(" Label")) != ""))

df = df.replace(float("inf"), None).replace(float("-inf"), None)

numeric_cols = [
    c for c, t in df.dtypes
    if t in ("double", "int", "bigint", "float")
]

for c in numeric_cols:
    df = df.withColumn(c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)))

null_counts_df = df.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in numeric_cols
])

print("\nNull counts (numeric columns):")
null_counts_df.show(truncate=False)


df = df.dropna(subset=numeric_cols)


dropped = set()
numeric_cols_only = [
    c for c, t in df.dtypes
    if t in ("double", "int", "bigint", "float")
]
cols = numeric_cols_only

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        c1, c2 = cols[i], cols[j]
        if c1 in dropped or c2 in dropped:
            continue
        mismatch = df.filter(
            (F.col(c1).isNull() != F.col(c2).isNull()) |
            (F.col(c1).isNotNull() & F.col(c2).isNotNull() & (F.col(c1) != F.col(c2)))
        ).limit(1).count()
        if mismatch == 0:
            df = df.drop(c2)
            dropped.add(c2)

if dropped:
    print("\nDropped duplicate columns:")
    for c in sorted(dropped):
        print(c)
else:
    print("\nNo duplicate columns found")


after_rows = df.count()
print("\nRows before cleaning:", before_rows)
print("Rows after cleaning:", after_rows)
print("Rows dropped:", before_rows - after_rows)


inv_rename_map = {v: k for k, v in rename_map.items()}
for new, old in inv_rename_map.items():
    if new in df.columns and new not in dropped:
        df = df.withColumnRenamed(new, old)


output_path = os.path.join(output_dir_path, "1B_clean_cicids2017.csv")

df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


spark.stop()
