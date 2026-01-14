import os
from pyspark.sql import SparkSession, functions as F, Window
from utils.schema import CIC_IDS_2017_T_FULL_CLEAN_SCHEMA


input_dir_path = os.path.join("data", "traffic_labelled", "1B_clean_cic_ids_t_2017")
output_dir_path = os.path.join("data", "traffic_labelled", "2B_preprocessed_logbert")
input_path = os.path.join(input_dir_path, "1B_clean_cic_ids_t_2017.csv")

spark = (
    SparkSession.builder
    .appName("CIC_IDS_T_2017 SEQUENCE BUILDER")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

MAX_LEN = 256
MIN_LEN = 5
WINDOW_SECONDS = 300

def bucketize_fixed(col_name, edges, prefix):
    expr = None
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        cond = (F.col(col_name) >= F.lit(lo)) & (F.col(col_name) < F.lit(hi))
        expr = F.when(cond, F.lit(f"{prefix}B{i}")) if expr is None else expr.when(cond, F.lit(f"{prefix}B{i}"))
    return expr.otherwise(F.lit(f"{prefix}B{len(edges)-1}"))

DUR_EDGES = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
PKT_EDGES = [0, 1, 2, 5, 10, 20, 50, 100, 1_000]
BYTES_EDGES = [0, 1, 64, 512, 4_096, 65_536, 1_048_576, 10_485_760]

df = (
    spark.read.format("csv")
    .schema(CIC_IDS_2017_T_FULL_CLEAN_SCHEMA)
    .option("header", True)
    .option("sep", ",")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .load(input_path)
)

df = df.select([F.col(c).alias(c.strip()) for c in df.columns])

df = df.withColumn("Timestamp", F.date_format(F.col("Timestamp"), "yyyy-MM-dd HH:mm:ss"))
df = df.withColumn("ts", F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss"))

df = df.withColumn("y", F.when(F.col("Label") == "BENIGN", F.lit(0)).otherwise(F.lit(1)))
df = df.withColumn("window_id", F.floor(F.unix_timestamp("ts") / F.lit(WINDOW_SECONDS)))

df = df.withColumn(
    "entity",
    F.concat_ws(
        "::",
        F.col("Source IP"),
        F.col("Destination Port").cast("string"),
        F.col("Protocol").cast("string"),
    ),
)

df = df.withColumn(
    "seq_id",
    F.concat_ws("_", F.col("day"), F.col("window_id").cast("string"), F.col("entity")),
)

base_tok = F.concat_ws("_", F.col("Protocol").cast("string"), F.col("Destination Port").cast("string"))

dur_bin = bucketize_fixed("Flow Duration", DUR_EDGES, "DUR_")
fwdp_bin = bucketize_fixed("Total Fwd Packets", PKT_EDGES, "FWD_")
bwdp_bin = bucketize_fixed("Total Backward Packets", PKT_EDGES, "BWD_")

df = df.withColumn("BYTES_TOTAL", F.col("Total Length of Fwd Packets") + F.col("Total Length of Bwd Packets"))
byt_bin = bucketize_fixed("BYTES_TOTAL", BYTES_EDGES, "BYT_")

df = df.withColumn("token", F.concat_ws("|", base_tok, dur_bin, fwdp_bin, bwdp_bin, byt_bin))

seqs = (
    df.groupBy("seq_id", "day", "window_id", "entity")
    .agg(
        F.max("y").alias("seq_y"),
        F.collect_list(F.struct(F.col("ts").alias("ts"), F.col("token").alias("tok"))).alias("tok_structs"),
    )
    .withColumn("tok_sorted", F.sort_array(F.col("tok_structs"), asc=True))
    .withColumn("tokens_full", F.expr("transform(tok_sorted, x -> x.tok)"))
    .drop("tok_structs", "tok_sorted")
)

seqs = seqs.withColumn("tokens", F.expr(f"slice(tokens_full, 1, {MAX_LEN})")).drop("tokens_full")

seqs = seqs.withColumn("seq_len", F.size("tokens"))
seqs = seqs.filter(F.col("seq_len") >= F.lit(MIN_LEN))

split_key = F.pmod(
    F.abs(
        F.hash(
            F.col("day"),
            F.col("seq_y"),
            F.col("seq_id")
        )
    ),
    F.lit(100)
)

seqs = seqs.withColumn(
    "split",
    F.when(split_key < 70, F.lit("train"))
     .when(split_key < 85, F.lit("val"))
     .otherwise(F.lit("test"))
)

out_seq = os.path.join(output_dir_path, "sequences")
out_meta = os.path.join(output_dir_path, "sequences_meta")

(
    seqs.select("seq_id", "day", "window_id", "entity", "split", "seq_y", "seq_len", "tokens")
    .write.mode("overwrite")
    .partitionBy("split", "day")
    .parquet(out_seq)
)

(
    seqs.select("seq_id", "day", "window_id", "entity", "split", "seq_y", "seq_len")
    .write.mode("overwrite")
    .parquet(out_meta)
)

spark.stop()
