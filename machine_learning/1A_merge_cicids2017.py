import os
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, when
from utils.schema import CIC_IDS_2017_M_SCHEMA


input_dir_path = os.path.join("..", "data", "machine_learning", "cic_ids_2017_m_raw")
output_dir_path = os.path.join("..", "data", "machine_learning", "1A_merge_cic_ids_m_2017")


spark = SparkSession.builder \
    .appName("CIC_IDS_M_2017 Merge") \
    .master("local[*]") \
    .getOrCreate()


csv_files = [
    f for f in os.listdir(input_dir_path)
    if f.lower().endswith(".csv")
]


dfs = []
row_counts = []


for file_name in csv_files:
    input_file_path = os.path.join(input_dir_path, file_name)

    df = spark.read.format("csv") \
        .schema(CIC_IDS_2017_M_SCHEMA) \
        .option("header", "false") \
        .load(input_file_path)

    df = df.filter(df[" Destination Port"].isNotNull())
    count_rows = df.count()
    row_counts.append((file_name, count_rows))

    df = df.withColumn(
        "day",
        when(lit(file_name).contains("Monday"), lit("Monday"))
        .when(lit(file_name).contains("Tuesday"), lit("Tuesday"))
        .when(lit(file_name).contains("Wednesday"), lit("Wednesday"))
        .when(lit(file_name).contains("Thursday"), lit("Thursday"))
        .when(lit(file_name).contains("Friday"), lit("Friday"))
        .otherwise(lit("Unknown"))
    )

    df = df.withColumn(
        "time_of_day",
        when(lit(file_name).contains("Morning"), lit("Morning"))
        .when(lit(file_name).contains("Afternoon"), lit("Afternoon"))
        .otherwise(lit("AllDay"))
    )

    dfs.append(df)


print("\nRows per file (before union):")
for name, cnt in row_counts:
    print(f"{name}: {cnt}")


final_df = reduce(lambda a, b: a.unionByName(b), dfs)


total_rows = final_df.count()
print(f"\nTotal rows after union: {total_rows}")


output_path = os.path.join(output_dir_path, "1A_merge_cic_ids_m_2017.csv")

final_df.coalesce(1) \
    .write \
    .option("header", True) \
    .mode("overwrite") \
    .csv(output_path)


spark.stop()
