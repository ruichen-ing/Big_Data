from pyspark.sql import SparkSession, DataFrame
from pyspark import TaskContext
from pyspark.sql.types import StructType, IntegerType
import pyspark.sql.functions as F
import random

TAU = 3000
HASH_VALUE_RANGE = 3000


def get_spark():
    global PARTITION_NUM
    return SparkSession.builder\
        .config("spark.sql.debug.maxToStringFields", 128)\
        .config("spark.sql.shuffle.partitions", PARTITION_NUM)\
        .config('spark.executor.memory', '4g')\
        .config('spark.sql.autoBroadcastJoinThreshold', "10m")\
        .getOrCreate()


def main():
    global spark
    spark = get_spark()

    def op_batch(window: DataFrame, id: int):
        window = window.filter(window.batch_id.between(id-29, id))
        window = window.groupBy("sid", "hash_func", "hash_val").sum(
            "count").withColumnRenamed("sum(count)", "count")
        left = window.alias("left")\
            .withColumnRenamed("sid", "sid1")\
            .withColumnRenamed("count", "count1")
        right = window.alias("right")\
            .withColumnRenamed("sid", "sid2")\
            .withColumnRenamed("count", "count2")
        window = left.join(right, [left.sid1 > right.sid2, left.hash_func == right.hash_func,
                           left.hash_val == right.hash_val], "inner").drop(right.hash_func).drop(right.hash_val)
        window = window.withColumn("product", window.count1 * window.count2)
        window = window.groupBy("sid1", "sid2", "hash_func").sum(
            "product")  # sum(product)
        window = window.groupBy("sid1", "sid2").min(
            "sum(product)").withColumnRenamed("min(sum(product))", "similarity")
        window = window.filter(window.similarity > TAU)
        nonlocal row_processed, sq
        info = sq.lastProgress
        if info:
            num = int(info["sources"][0]["numInputRows"])
            row_processed += num
        print("@@@%d, %d, %d" %
              (row_processed, window.count(), id*2), flush=True)
    row_processed = 0
    random_hash_seed = [(random.getrandbits(31),) for _ in range(5)]
    df_hash_fun = spark.createDataFrame(
        random_hash_seed, StructType().add("hash_func", IntegerType(), True))\
        .cache()
    df = spark.readStream\
        .format("socket")\
        .option("host", HOSTNAME)\
        .option("port", 9000)\
        .load()
    df = df\
        .withColumn("sid", F.split(df.value, ',').getItem(0).cast(IntegerType()))\
        .withColumn("ip", F.split(df.value, ',').getItem(1).cast(IntegerType()))\
        .drop("value")
    batchid_col = F.udf(lambda: int(TaskContext.get().getLocalProperty(
        "streaming.sql.batchId")), IntegerType())
    df = df.withColumn("batch_id", batchid_col())
    df = df.crossJoin(df_hash_fun)
    df = df.withColumn("hash_val", F.hash(
        df["ip"]).bitwiseXOR(df["hash_func"]) % HASH_VALUE_RANGE)
    df = df.select(
        "sid", "hash_func", "hash_val", "batch_id")
    df = df.groupBy(
        df.sid, df.hash_func, df.hash_val, df.batch_id).count()
    sq = df.writeStream\
        .trigger(processingTime="2 seconds")\
        .outputMode("complete")\
        .foreachBatch(op_batch)\
        .start()
    sq.awaitTermination(300)


PARTITION_NUM = 12 # 24 for cluster
HOSTNAME = "localhost" # "stream-host" for cluster

if __name__ == "__main__":
    main()
