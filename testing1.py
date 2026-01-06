import os

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType, DoubleType, TimestampType
)

from functools import reduce
import operator

input_file_path = os.path.join("..", "raw_data", "MachineLearningCSV.csv")

spark = (
    SparkSession.builder
    .appName("CSV to Dataset")
    .master("local[*]")
    .getOrCreate()
)



schema = StructType([
    StructField(" Destination Port", IntegerType(), True),

    StructField(" Flow Duration", DoubleType(), True),
    StructField(" Total Fwd Packets", DoubleType(), True),
    StructField(" Total Backward Packets", DoubleType(), True),
    StructField("Total Length of Fwd Packets", DoubleType(), True),
    StructField(" Total Length of Bwd Packets", DoubleType(), True),
    StructField(" Fwd Packet Length Max", DoubleType(), True),
    StructField(" Fwd Packet Length Min", DoubleType(), True),
    StructField(" Fwd Packet Length Mean", DoubleType(), True),
    StructField(" Fwd Packet Length Std", DoubleType(), True),
    StructField("Bwd Packet Length Max", DoubleType(), True),
    StructField(" Bwd Packet Length Min", DoubleType(), True),
    StructField(" Bwd Packet Length Mean", DoubleType(), True),
    StructField(" Bwd Packet Length Std", DoubleType(), True),
    StructField("Flow Bytes/s", DoubleType(), True),
    StructField(" Flow Packets/s", DoubleType(), True),
    StructField(" Flow IAT Mean", DoubleType(), True),
    StructField(" Flow IAT Std", DoubleType(), True),
    StructField(" Flow IAT Max", DoubleType(), True),
    StructField(" Flow IAT Min", DoubleType(), True),
    StructField("Fwd IAT Total", DoubleType(), True),
    StructField(" Fwd IAT Mean", DoubleType(), True),
    StructField(" Fwd IAT Std", DoubleType(), True),
    StructField(" Fwd IAT Max", DoubleType(), True),
    StructField(" Fwd IAT Min", DoubleType(), True),
    StructField("Bwd IAT Total", DoubleType(), True),
    StructField(" Bwd IAT Mean", DoubleType(), True),
    StructField(" Bwd IAT Std", DoubleType(), True),
    StructField(" Bwd IAT Max", DoubleType(), True),
    StructField(" Bwd IAT Min", DoubleType(), True),
    StructField("Fwd PSH Flags", DoubleType(), True),
    StructField(" Bwd PSH Flags", DoubleType(), True),
    StructField(" Fwd URG Flags", DoubleType(), True),
    StructField(" Bwd URG Flags", DoubleType(), True),
    StructField(" Fwd Header Length40", DoubleType(), True),
    StructField(" Bwd Header Length", DoubleType(), True),
    StructField("Fwd Packets/s", DoubleType(), True),
    StructField(" Bwd Packets/s", DoubleType(), True),
    StructField(" Min Packet Length", DoubleType(), True),
    StructField(" Max Packet Length", DoubleType(), True),
    StructField(" Packet Length Mean", DoubleType(), True),
    StructField(" Packet Length Std", DoubleType(), True),
    StructField(" Packet Length Variance", DoubleType(), True),
    StructField("FIN Flag Count", DoubleType(), True),
    StructField(" SYN Flag Count", DoubleType(), True),
    StructField(" RST Flag Count", DoubleType(), True),
    StructField(" PSH Flag Count", DoubleType(), True),
    StructField(" ACK Flag Count", DoubleType(), True),
    StructField(" URG Flag Count", DoubleType(), True),
    StructField(" CWE Flag Count", DoubleType(), True),
    StructField(" ECE Flag Count", DoubleType(), True),
    StructField(" Down/Up Ratio", DoubleType(), True),
    StructField(" Average Packet Size", DoubleType(), True),
    StructField(" Avg Fwd Segment Size", DoubleType(), True),
    StructField(" Avg Bwd Segment Size", DoubleType(), True),
    StructField(" Fwd Header Length61", DoubleType(), True),
    StructField("Fwd Avg Bytes/Bulk", DoubleType(), True),
    StructField(" Fwd Avg Packets/Bulk", DoubleType(), True),
    StructField(" Fwd Avg Bulk Rate", DoubleType(), True),
    StructField(" Bwd Avg Bytes/Bulk", DoubleType(), True),
    StructField(" Bwd Avg Packets/Bulk", DoubleType(), True),
    StructField("Bwd Avg Bulk Rate", DoubleType(), True),
    StructField("Subflow Fwd Packets", DoubleType(), True),
    StructField(" Subflow Fwd Bytes", DoubleType(), True),
    StructField(" Subflow Bwd Packets", DoubleType(), True),
    StructField(" Subflow Bwd Bytes", DoubleType(), True),
    StructField("Init_Win_bytes_forward", DoubleType(), True),
    StructField(" Init_Win_bytes_backward", DoubleType(), True),
    StructField(" act_data_pkt_fwd", DoubleType(), True),
    StructField(" min_seg_size_forward", DoubleType(), True),
    StructField("Active Mean", DoubleType(), True),
    StructField(" Active Std", DoubleType(), True),
    StructField(" Active Max", DoubleType(), True),
    StructField(" Active Min", DoubleType(), True),
    StructField("Idle Mean", DoubleType(), True),
    StructField(" Idle Std", DoubleType(), True),
    StructField(" Idle Max", DoubleType(), True),
    StructField(" Idle Min", DoubleType(), True),

    StructField(" Label", StringType(), True),
])



df = (
    spark.read
      .option("header", True)
      .option("sep", ",")
      .option("mode", "FAILFAST")
      .option("timestampFormat", "d/M/yyyy H:mm")
      .schema(schema)
      .csv(input_file_path)
)

df.printSchema()

all_null = reduce(operator.and_, [F.col(c).isNull() for c in df.columns])

df = df.filter(~all_null)

print(df.count())


df = df.orderBy(F.rand()).limit(100)

output_file = os.path.join("MachineLearningCSV.csv")

(
    df.coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(output_file)
)

spark.stop()
