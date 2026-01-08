import os
from functools import reduce
import operator

from pyspark.sql import SparkSession, functions as F
from utils.schema import CIC_IDS_2017_T_FULL_SCHEMA


input_dir_path = os.path.join( "data", "traffic_labelled", "1A_merge_cic_ids_t_2017")
output_dir_path = os.path.join( "data", "traffic_labelled", "1B_clean_cic_ids_t_2017")


spark = SparkSession.builder \
    .appName("CIC_IDS_T_2017 Cleaning 1B") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.codegen.wholeStage", "false")


input_path = os.path.join(input_dir_path, "1A_merged_cic_ids_t_2017.csv")

df = (
    spark.read
        .format("csv")
        .schema(CIC_IDS_2017_T_FULL_SCHEMA)
        .option("header", True)
        .option("sep", ",")
        .load(input_path)
)

before_rows = df.count()


dot_cols = [c for c in df.columns if "." in c]
for c in dot_cols:
    df = df.withColumnRenamed(c, c.replace(".", "_"))


all_null = reduce(operator.and_, [F.col(c).isNull() for c in df.columns])
df = df.filter(~all_null)

df = df.filter(F.col(" Label").isNotNull() & (F.trim(F.col(" Label")) != ""))

df = df.replace(float("inf"), None).replace(float("-inf"), None)

numeric_cols = [c for c, t in df.dtypes if t in ("double", "int", "bigint", "float")]

for c in numeric_cols:
    df = df.withColumn(c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)))

df = df.dropna(subset=numeric_cols)


col_a = " Fwd Header Length"
col_b = " Fwd Header Length_1"

if col_a in df.columns and col_b in df.columns:
    mismatch = df.filter(
        (F.col(col_a).isNull() != F.col(col_b).isNull()) |
        (F.col(col_a).isNotNull() & F.col(col_b).isNotNull() & (F.col(col_a) != F.col(col_b)))
    ).limit(1).count()

    if mismatch == 0:
        df = df.drop(col_b)
        print(f"\nDropped duplicate column: {col_b}")
    else:
        print(f"\nColumns are not identical: '{col_a}' vs '{col_b}'")
else:
    print("\nDuplicate-check columns not found")


after_rows = df.count()
print("\nRows before cleaning:", before_rows)
print("Rows after cleaning:", after_rows)
print("Rows dropped:", before_rows - after_rows)


output_path = os.path.join(output_dir_path, "1B_clean_cic_ids_t_2017.csv")

df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


spark.stop()
