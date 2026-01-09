import os
from functools import reduce
import operator

from pyspark.sql import SparkSession, functions as F, Window
from utils.schema import CIC_IDS_2017_T_FULL_SCHEMA, CIC_IDS_2017_T_FULL_CLEAN_SCHEMA

input_dir_path = os.path.join("..", "data", "traffic_labelled", "1B_clean_cic_ids_t_2017")
# output_dir_path = os.path.join("..", "data", "traffic_labelled", "1B_clean_cic_ids_t_2017")
#

spark = SparkSession.builder \
    .appName("CIC_IDS_T_2017 PREPROCESSING 2A") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


input_path = os.path.join(input_dir_path, "1B_clean_cic_ids_t_2017.csv")

df = (
    spark.read
        .format("csv")
        .schema(CIC_IDS_2017_T_FULL_CLEAN_SCHEMA)
        .option("header", True)
        .option("sep", ",")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .load(input_path)
)

df = df.withColumn(" Timestamp", F.date_format(F.col(" Timestamp"), "yyyy-MM-dd HH:mm:ss"))

df = df.withColumn(
    "y",
    F.when(F.col(" Label") == "BENIGN", F.lit(0)).otherwise(F.lit(1))
)

df = df.withColumn("window_id", F.floor(F.unix_timestamp(" Timestamp") / F.lit(300)))

w = Window.partitionBy("day").orderBy(" Timestamp")

df = df.withColumn("node_id", F.row_number().over(w) - 1)

df.show(n=20, truncate=False)

spark.stop()
