import os
from functools import reduce
import operator

from pyspark.sql import SparkSession, functions as F
from utils.schema import CIC_IDS_2017_T_FULL_SCHEMA


input_dir_path = os.path.join("..", "data", "traffic_labelled", "1B_clean_cic_ids_t_2017")
output_dir_path = os.path.join("..", "data", "traffic_labelled", "A_sample")


spark = SparkSession.builder \
    .appName("CIC_IDS_T_2017 Cleaning 1B") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

input_path = os.path.join(input_dir_path, "1B_clean_cic_ids_t_2017.csv")

df = (
    spark.read
        .format("csv")
        .schema(CIC_IDS_2017_T_FULL_SCHEMA)
        .option("header", True)
        .option("sep", ",")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .load(input_path)
)

df = df.orderBy(F.rand()).limit(1000)

output_path = os.path.join(output_dir_path, "A_sample.csv")

df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)


spark.stop()
