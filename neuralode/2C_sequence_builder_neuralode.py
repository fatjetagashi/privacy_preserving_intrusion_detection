import os
from pyspark.sql import SparkSession, functions as F, Window
from utils.schema import CIC_IDS_2017_T_FULL_CLEAN_SCHEMA

WINDOW_SECONDS = 60
MIN_LEN = 5
MAX_LEN_BUILD = 2000

input_dir_path = os.path.join("..", "data", "traffic_labelled", "1B_clean_cic_ids_t_2017")
output_dir_path = os.path.join("..", "data", "traffic_labelled", "2C_preprocessed_neuralode")

input_path = os.path.join(input_dir_path, "1B_clean_cic_ids_t_2017.csv")

spark = (
    SparkSession.builder
    .appName("CIC_IDS_T_2017 NeuralODE Sequence Builder")
    .master("local[4]")
    .config("spark.driver.memory", "12g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = (
    spark.read.format("csv")
    .schema(CIC_IDS_2017_T_FULL_CLEAN_SCHEMA)
    .option("header", True)
    .option("sep", ",")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .load(input_path)
)

df = df.select([F.col(c).alias(c.strip()) for c in df.columns])

df = df.withColumn("ts", F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss"))
df = df.withColumn("ts_unix", F.unix_timestamp("ts").cast("long"))

df = df.withColumn("y", F.when(F.col("Label") == "BENIGN", F.lit(0)).otherwise(F.lit(1)))

df = df.withColumn("window_id", F.floor(F.col("ts_unix") / F.lit(WINDOW_SECONDS)))

df = df.withColumn(
    "entity",
    F.concat_ws(
        "::",
        F.col("Destination IP"),
        F.col("Destination Port").cast("string"),
        F.col("Protocol").cast("string"),
    )
)

df = df.withColumn("seq_id", F.concat_ws("_", F.col("day"), F.col("window_id").cast("string"), F.col("entity")))

exclude = {"Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "ts",
           "ts_unix", "Label", "day", "time_of_day", "window_id", "entity", "seq_id", "y"}

feature_cols = [c for c in df.columns if c not in exclude]

df = df.withColumn("x", F.array([F.col(c).cast("double") for c in feature_cols]))

seqs = (
    df.groupBy("seq_id","day","window_id","entity")
      .agg(
          F.max("y").alias("seq_y"),
          F.collect_list(F.struct(
              F.col("ts_unix").alias("t"),
              F.col("x").alias("x"),
              F.col("y").alias("y")
          )).alias("events")
      )
      .withColumn("events_sorted", F.sort_array(F.col("events"), asc=True))
      .drop("events")
)

seqs = seqs.withColumn("t_full", F.expr("transform(events_sorted, e -> e.t)")) \
           .withColumn("x_full", F.expr("transform(events_sorted, e -> e.x)")) \
           .withColumn("y_full", F.expr("transform(events_sorted, e -> e.y)")) \
           .drop("events_sorted")

seqs = seqs.withColumn(
    "t",
    F.expr(f"slice(t_full, greatest(size(t_full)-{MAX_LEN_BUILD}+1,1), {MAX_LEN_BUILD})")
).withColumn(
    "x",
    F.expr(f"slice(x_full, greatest(size(x_full)-{MAX_LEN_BUILD}+1,1), {MAX_LEN_BUILD})")
).withColumn(
    "y_seq",
    F.expr(f"slice(y_full, greatest(size(y_full)-{MAX_LEN_BUILD}+1,1), {MAX_LEN_BUILD})")
).drop("t_full","x_full","y_full")

seqs = seqs.withColumn("seq_len", F.size("t")).filter(F.col("seq_len") >= F.lit(MIN_LEN))

split_key = F.pmod(F.abs(F.hash(F.col("seq_id"))), F.lit(100))

seqs = seqs.withColumn(
    "split",
    F.when(split_key < 70, F.lit("train"))
     .when(split_key < 85, F.lit("val"))
     .otherwise(F.lit("test"))
)

out_seq = os.path.join(output_dir_path, "sequences")
out_meta = os.path.join(output_dir_path, "sequences_meta")

(
    seqs.select("seq_id","day","window_id","entity","split","seq_y","seq_len","t","x","y_seq")
        .write.mode("overwrite")
        .partitionBy("split","day")
        .parquet(out_seq)
)

(
    seqs.select("seq_id","day","window_id","entity","split","seq_y","seq_len")
        .write.mode("overwrite")
        .parquet(out_meta)
)

spark.stop()