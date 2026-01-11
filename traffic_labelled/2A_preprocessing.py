import os
from functools import reduce
import operator

from pyspark.sql import SparkSession, functions as F, Window
from utils.schema import CIC_IDS_2017_T_FULL_SCHEMA, CIC_IDS_2017_T_FULL_CLEAN_SCHEMA

input_dir_path = os.path.join("..", "data", "traffic_labelled", "1B_clean_cic_ids_t_2017")
output_dir_path = os.path.join("..", "data", "traffic_labelled", "2A_preprocessed_gnn")


spark = SparkSession.builder \
    .appName("CIC_IDS_T_2017 PREPROCESSING 2A") \
    .master("local[4]") \
    .config("spark.driver.memory", "12g")   \
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

df = df.select([F.col(c).alias(c.strip()) for c in df.columns])

df = df.withColumn("Timestamp", F.date_format(F.col("Timestamp"), "yyyy-MM-dd HH:mm:ss"))

df = df.withColumn(
    "y",
    F.when(F.col("Label") == "BENIGN", F.lit(0)).otherwise(F.lit(1))
)

df = df.withColumn("window_id", F.floor(F.unix_timestamp("Timestamp") / F.lit(300)))
df = df.withColumn("graph_id", F.concat_ws("_", F.col("day"), F.col("window_id")))

w = Window.partitionBy("graph_id").orderBy(F.col("Timestamp"), F.col("Flow ID"))
df = df.withColumn("node_id", F.row_number().over(w) - 1)

def chain_edges(base_df, group_cols, k=3):
    w = Window.partitionBy(*group_cols).orderBy("Timestamp", "node_id")
    out = None
    for i in range(1, k + 1):
        tmp = (base_df
               .withColumn("dst", F.lead("node_id", i).over(w))
               .where(F.col("dst").isNotNull())
               .select("graph_id", F.col("node_id").alias("src"), "dst"))
        out = tmp if out is None else out.unionByName(tmp)
    return out

e_src = chain_edges(df, ["graph_id", "Source IP"], k=5)
e_dst = chain_edges(df, ["graph_id", "Destination IP"], k=5)
e_svc = chain_edges(df, ["graph_id", "Destination Port", "Protocol"], k=5)

edges = e_src.unionByName(e_dst).unionByName(e_svc)
edges_rev = edges.select("graph_id", F.col("dst").alias("src"), F.col("src").alias("dst"))

edges = edges.unionByName(edges_rev).dropDuplicates(["graph_id", "src", "dst"])


non_features = {"graph_id","day","time_of_day","window_id","node_id","y","Label","Flow ID", "Source Port",
                "Timestamp","Source IP","Destination IP","Destination Port","Protocol"}
feature_cols = [c for c, t in df.dtypes if c not in non_features and t in ("double","int","bigint","float")]

graphs = (df.groupBy("graph_id","day","window_id")
            .agg(F.max("y").alias("graph_y")))

graphs = graphs.withColumn("r", F.rand(42))

w_split = Window.partitionBy("day","graph_y").orderBy("r")
graphs = graphs.withColumn("p", F.percent_rank().over(w_split))

graphs = graphs.withColumn(
    "split",
    F.when(F.col("p") < 0.70, F.lit("train"))
     .when(F.col("p") < 0.85, F.lit("val"))
     .otherwise(F.lit("test"))
)

nodes = (df.join(graphs.select("graph_id","split"), on="graph_id", how="inner")
           .select("graph_id","day","window_id","split","node_id","y", *feature_cols))

edge_meta = graphs.select("graph_id","day","window_id","split")
edges = edges.join(edge_meta, on="graph_id", how="inner")

output_path_nodes = os.path.join(output_dir_path, "nodes")
output_path_edges = os.path.join(output_dir_path, "edges")
output_path_graphs_meta = os.path.join(output_dir_path, "graphs_meta")

nodes.write.mode("overwrite").partitionBy("split","day").parquet(output_path_nodes)
edges.write.mode("overwrite").partitionBy("split","day").parquet(output_path_edges)
graphs.write.mode("overwrite").parquet(output_path_graphs_meta)

spark.stop()
