import time
from math import sqrt
from datetime import date, timedelta
from operator import index
from typing import Iterator, Tuple, List, Dict, Iterable
from numpy import partition
from pyspark import Row
from pyspark.sql import SparkSession
from pyspark import SparkContext, RDD, AccumulatorParam, accumulators
from pyspark.sql.types import StructType, StringType, FloatType, IntegerType, DateType
from pyspark import StorageLevel
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import time
import sys
import os


class listAccumulator(AccumulatorParam):
    def zero(self, value: List):
        return []

    def addInPlace(self, value1: List, value2: Dict) -> List:
        return value1.append(value2)


class Preparer:

    def __enviorn(self):
        """ Private: Get enviornment info
        """
        p = sys.argv[0]
        self.dir = os.path.dirname(p)

    def __init__(self, spark: SparkSession, path="sample.csv", print_debug=False):
        """ Create a spark session and read data from file.

        Args:
            path (str, optional): path of data file. Defaults to "sample.csv".
            partition_num (int, optional): partition_num. Defaults to 12.
            print_debug (bool, optional): whether to print debug info, may 
            influence performence. Defaults to False.
        """
        self.__enviorn()
        self.data_path = path
        # https://stackoverflow.com/questions/25600993/spark-indefinite-waiting-with-asked-to-send-map-output-locations-for-shuffle
        self.spark = spark
        self.stock_schema = StructType()\
            .add("stock", StringType(), False)\
            .add("date", DateType(), False)\
            .add("price", FloatType(), False)
        self.print_debug = print_debug
        self.df = self.read()
        self.__show_desc(self.df)

    def read(self) -> DataFrame:
        """ read data from data src and partition by stock name.

        Returns:
            DataFrame: partitioned dataframe
        """
        # The first schema is defferent from the later ued ones
        schema = StructType()\
            .add("stock", StringType(), False)\
            .add("date", DateType(), False)\
            .add("price", FloatType(), False)\
            .add("volume", IntegerType(), False)
        df = self.spark.read.format('csv')\
            .schema(schema)\
            .option("dateFormat", "MM/dd/yyyy") \
            .load(self.data_path)
        df = df.drop("volume")  # we dont need this
        df = self.__make_partition(df)
        self.df = df
        return df

    def __make_partition(self, df: DataFrame):
        df = df\
            .repartition(SPARK_PARTITION_NUM, df.stock)\
            .persist(StorageLevel.MEMORY_AND_DISK)
        return df

    def __get_week_days(self):
        """ A list of all dates in date range 2016-1-1 to 2020-12-31

        Returns:
            _type_: _description_
        """
        start_date = date(2016, 1, 1)
        end_date = date(2020, 12, 31)
        day = start_date
        ret = []
        while day <= end_date:
            if day.weekday() not in (5, 6):
                ret.append({"date": day})
            day = day + timedelta(days=1)
        return ret

    def prev_weekday(d: date):
        if d.weekday() == 0:
            nd = d + timedelta(days=-3)
        else:
            nd = d + timedelta(days=-1)
        return nd

    def __show_desc(self, d: DataFrame):
        if self.print_debug:
            d.describe().show()

    def clean(self) -> DataFrame:
        """ Clean the data with following rules.\n
        a) remove all duplications.\n
        b) remove all rows not in time_series range 2016-1-1 to 2020-12-31\n
        c) time align all the data. Fill by copy previous date's data. If can not fill
        the data in this way, then drop entire time series.

        Returns:
            DataFrame: cleaned data
        """
        # Duplication remove
        df = self.df
        df = df.dropDuplicates(['stock', 'date'])
        # filter date to only weekdays in range
        df = df.filter(F.dayofweek(df.date).isin([2, 3, 4, 5, 6]))
        df = df.filter(df.date.between(date(2016, 1, 1), date(2020, 12, 31)))
        # get full_df set: {stock x time_series}
        stock_name_df = df.select(df.stock).distinct()
        time_series = self.spark.createDataFrame(self.__get_week_days())
        full_df = time_series.crossJoin(stock_name_df)
        # filter out lost dates for further operation
        diff = full_df.subtract(df.select(df.date, df.stock))
        udf_prev_weekday = F.udf(Preparer.prev_weekday, DateType())
        diff_prev = diff.withColumn("prev", udf_prev_weekday(diff.date))
        df_rename = df.select(df.date.alias("df_date"),
                              df.stock.alias("df_stock"), df.price)
        # ==== Repeat this part to increase the fault tolerance:
        # for once, means any continuous missing of dates will not be tolerated.
        diff = diff_prev\
            .join(df_rename, (diff_prev.stock == df_rename.df_stock) & (diff_prev.prev == df_rename.df_date), "left")\
            .drop("df_stock").drop("df_date")
        # ==== END repeat
        diff = diff.drop("prev")
        # union the origin data and diff data together, which are the desired time-aligned data
        # though there are some price/volume vavlue leave blank
        df = df.union(diff.select("stock", "date", "price"))
        self.__show_desc(df)
        # Now delete those stocks with null values
        # which means there are values can not be complete (due to multiple reasons)
        stock_with_null = df.filter(
            df.price.isNull()).select(df.stock).distinct()
        stock_with_null = stock_with_null.select(
            stock_with_null.stock.alias("swn_stock"))
        self.__show_desc(stock_with_null)
        # https://stackoverflow.com/questions/42545788/pyspark-match-the-values-of-a-dataframe-column-against-another-dataframe-column
        df = df.subtract(df.join(stock_with_null, df.stock ==
                         stock_with_null.swn_stock, "leftsemi"))
        df = df.orderBy("stock", "date")
        self.__show_desc(df)
        self.df = df
        return df

    def limit_stock_num(self, num=1000):
        """Select number of stocks from data.

        Args:
            num (int, optional): number of stocks. Defaults to 1000.

        Returns:
            DataFrame: df
        """
        df = self.df
        stocks = df.select(df.stock).distinct()
        stocks = stocks.withColumn("idx", F.split(
            stocks.stock, '\.').getItem(0).cast(IntegerType()))
        stocks = stocks.orderBy(stocks.idx).limit(num)
        df = df.join(stocks, df.stock == stocks.stock, "leftsemi")
        return df


class Calculator:

    def __init__(self, spark, stock_rdd: RDD, wiki_path="wiki_sample.csv") -> None:
        if spark:
            self.spark = spark
        else:
            self.spark = SparkSession.builder.getOrCreate()
        self.stock_rdd = stock_rdd
        # Only use dataframe when reading data
        schema = StructType()\
            .add("topic", StringType(), False)\
            .add("date", DateType(), False)\
            .add("visit", IntegerType(), False)
        wiki_df = self.spark.read\
            .schema(schema)\
            .option("dateFormat", "MM/dd/yyyy")\
            .option("header", "true")\
            .csv(wiki_path)\
            .cache()
        self.wiki_count = wiki_df.select("topic").distinct().count()
        self.wiki_rdd = wiki_df.rdd

    def title_index(self):
        """ index wiki topics and stocks' names to integer to reduce join cost.
        """
        # wiki
        rdd = self.wiki_rdd
        index_rdd = rdd.map(lambda x: x["topic"])\
            .distinct()\
            .zipWithIndex()
        self.wiki_topic_index = index_rdd
        rdd = rdd.keyBy(lambda x: x["topic"])
        index_rdd = index_rdd.keyBy(lambda x: x[0])
        rdd = rdd.join(index_rdd).map(lambda x: x[1])
        rdd = rdd.map(lambda x: {
            "topic": x[1][1],
            "date": x[0]["date"],
            "visit": x[0]["visit"]
        })
        self.wiki_rdd = rdd
        # stock
        rdd = self.stock_rdd
        index_rdd = rdd.map(lambda x: x["stock"])\
            .distinct()\
            .zipWithIndex()
        self.stock_stock_index = index_rdd
        rdd = rdd.keyBy(lambda x: x["stock"])
        index_rdd = index_rdd.keyBy(lambda x: x[0])
        rdd = rdd.join(index_rdd).map(lambda x: x[1])
        rdd = rdd.map(lambda x: {
            "stock": x[1][1],
            "date": x[0]["date"],
            "price": x[0]["price"]
        })
        self.stock_rdd = rdd

    def report_lookup(self, rdd: RDD) -> RDD:
        # topic1
        index_rdd = self.wiki_topic_index.keyBy(lambda x: x[1])
        rdd = rdd.keyBy(lambda x: x["topic1"])
        rdd = rdd.join(index_rdd).map(lambda x: x[1])
        rdd = rdd.map(lambda x: {
            "similarity": x[0]["similarity"],
            "stock": x[0]["stock"],
            "topic1": x[1][0],
            "topic2": x[0]["topic2"]
        })
        # topic2
        rdd = rdd.keyBy(lambda x: x["topic2"])
        rdd = rdd.join(index_rdd).map(lambda x: x[1])
        rdd = rdd.map(lambda x: {
            "similarity": x[0]["similarity"],
            "stock": x[0]["stock"],
            "topic1": x[0]["topic1"],
            "topic2": x[1][0]
        })
        # stock
        index_rdd = self.stock_stock_index.keyBy(lambda x: x[1])
        rdd = rdd.keyBy(lambda x: x["stock"])
        rdd = rdd.join(index_rdd).map(lambda x: x[1])
        rdd = rdd.map(lambda x: {
            "similarity": x[0]["similarity"],
            "stock": x[1][0],
            "topic1": x[0]["topic1"],
            "topic2": x[0]["topic2"]
        })
        return rdd

    def date_index(self):
        """transform date object to int index, since all the series are time-aligned.
        Apllied both to stock rdd and wiki rdd.
        This is a performance consideration cuz date type comparasion cost much higher
        than interger.
        """
        def get_date_dic() -> List[dict]:
            """ A list of dic of all dates in date range 2016-1-1 to 2020-12-31 index
            to 0...1304

            Returns:
                List: list of dic, key=date, item=index
            """
            start_date = date(2016, 1, 1)
            end_date = date(2020, 12, 31)
            day = start_date
            idx = 0
            ret = []
            while day <= end_date:
                if day.weekday() not in (5, 6):
                    ret.append({"date": day, "idx": idx})
                idx += 1
                day = day + timedelta(days=1)
            return ret

        date_rdd = self.spark.sparkContext.parallelize(get_date_dic())
        date_rdd = date_rdd.keyBy(lambda row: row["date"]).partitionBy(
            SPARK_PARTITION_NUM).persist()
        rdd = self.wiki_rdd.keyBy(
            lambda row: row["date"]).partitionBy(SPARK_PARTITION_NUM)
        rdd = rdd.join(date_rdd)
        rdd = rdd.map(lambda x: x[1])
        rdd = rdd.map(
            lambda x: {"topic": x[0]["topic"], "date": x[1]["idx"], "visit": x[0]["visit"]})
        self.wiki_rdd = rdd
        rdd = self.stock_rdd.keyBy(
            lambda row: row["date"]).partitionBy(SPARK_PARTITION_NUM)
        rdd = rdd.join(date_rdd)
        rdd = rdd.map(lambda x: x[1])
        rdd = rdd.map(
            lambda x: {"stock": x[0]["stock"], "date": x[1]["idx"], "price": x[0]["price"]})
        self.stock_rdd = rdd

    def calc_wiki_comb(self):
        """Combination calculation of wiki data. Combine every two of them and report.
        """
        rdd = self.wiki_rdd
        rdd = rdd.keyBy(lambda x: x["date"])
        rdd = rdd.join(rdd)
        rdd = rdd.filter(lambda x: x[1][0]["topic"] > x[1][1]["topic"])
        rdd = rdd.map(lambda x: {
            "topic": (x[1][0]["topic"], x[1][1]["topic"]),
            "date": x[1][0]["date"],
            "mean": (x[1][0]["visit"]+x[1][1]["visit"])/2,
            "min": min(x[1][0]["visit"], x[1][1]["visit"]),
            "max": max(x[1][0]["visit"], x[1][1]["visit"])
        })
        self.wiki_agg_rdd = rdd

    def cosine_similarity(self):
        """calc cos simi using rdd
        """
        wiki = self.wiki_agg_rdd.keyBy(lambda x: x["date"])\
            .partitionBy(SPARK_PARTITION_NUM)
        stock = self.stock_rdd.keyBy(lambda x: x["date"])\
            .partitionBy(SPARK_PARTITION_NUM)
        rdd = stock.join(wiki, SPARK_PARTITION_NUM)
        rdd = rdd.map(lambda x: {
            "triple": (x[1][0]["stock"], x[1][1]["topic"][0], x[1][1]["topic"][1]),
            "x2": x[1][0]["price"] ** 2,
            "xy_mean": x[1][0]["price"] * x[1][1]["mean"],
            "xy_max": x[1][0]["price"] * x[1][1]["max"],
            "xy_min": x[1][0]["price"] * x[1][1]["min"],
            "y2_mean": x[1][1]["mean"] ** 2,
            "y2_min": x[1][1]["min"] ** 2,
            "y2_max": x[1][1]["max"] ** 2
        })
        rdd = rdd.keyBy(lambda x: x["triple"]).partitionBy(SPARK_PARTITION_NUM)
        rdd = rdd.reduceByKey(lambda x, y: {
            "triple": x["triple"],
            "x2": x["x2"] + y["x2"],
            "xy_mean": x["xy_mean"] + y["xy_mean"],
            "xy_max": x["xy_max"] + y["xy_max"],
            "xy_min": x["xy_min"] + y["xy_min"],
            "y2_mean": x["y2_mean"] + y["y2_mean"],
            "y2_min": x["y2_min"] + y["y2_min"],
            "y2_max": x["y2_max"] + y["y2_max"]
        })
        rdd = rdd.map(lambda x: {
            "triple": x[1]["triple"],
            "cos_mean": x[1]["xy_mean"] / (sqrt(x[1]["x2"]) * sqrt(x[1]["y2_mean"])),
            "cos_min": x[1]["xy_min"] / (sqrt(x[1]["x2"]) * sqrt(x[1]["y2_min"])),
            "cos_max": x[1]["xy_max"] / (sqrt(x[1]["x2"]) * sqrt(x[1]["y2_max"])),
        })
        self.cos_rdd = rdd

    def calc_wiki_comb_bd(self):
        def join_with_bd(x: Tuple):
            """This is equel to:
                rdd = rdd.join(rdd)
                rdd = rdd.filter(lambda x: x[1][0]["topic"] > x[1][1]["topic"])
                rdd = rdd.map(lambda x: {
                    "topic": (x[1][0]["topic"], x[1][1]["topic"]),
                    "date": x[1][0]["date"],
                    "mean": (x[1][0]["visit"]+x[1][1]["visit"])/2,
                    "min": min(x[1][0]["visit"], x[1][1]["visit"]),
                    "max": max(x[1][0]["visit"], x[1][1]["visit"])
                })
                rdd = rdd.keyBy(lambda x: x["date"])
            """
            res = []
            for row in wiki_bd.value.get(x[0], []):
                if row and row[0] > x[1][0]:
                    res.append((
                        x[0],
                        {
                            "topic": (x[1][0], row[0]),
                            "mean": (x[1][1]+row[1])/2,
                            "min": min(x[1][1], row[1]),
                            "max": max(x[1][1], row[1])
                        }
                    ))
            return res
        rdd = self.wiki_rdd
        rdd = rdd.keyBy(lambda x: x["topic"])
        rdd = rdd.partitionBy(self.wiki_count // WIKI_PER_PARTITION).cache()
        rdd = rdd.map(lambda x: (x[1]["date"], (x[1]["topic"], x[1]["visit"])))
        wiki_bd = self.spark.sparkContext.broadcast(
            rdd.groupByKey().collectAsMap())
        rdd = rdd.flatMap(join_with_bd, True)
        rdd.cache()
        self.wiki_agg_rdd_key_by = rdd

    def cosine_similarity_bd(self):
        """calc cos simi using rdd
        """
        def join_with_bd(x: Tuple):
            res = []
            wiki = {
                "topic": x[1]["topic"],
                "mean": x[1]["mean"],
                "min": x[1]["min"],
                "max": x[1]["max"],
            }
            for row in stock_bd.value.get(x[0], []):
                if row:
                    res.append((
                        x[0],  # key
                        (
                            {
                                "stock": row[0],  # stock
                                "price": row[1]  # price
                            },  # join lefter stock
                            wiki  # join righter wiki
                        )
                    ))
            return res
        wiki = self.wiki_agg_rdd_key_by
        # https://stackoverflow.com/questions/34053302/pyspark-and-broadcast-join-example
        # to save transfer bandwidth, map to (key, (dict->tuple))
        stock = self.stock_rdd.map(lambda x: (
            x["date"], (x["stock"], x["price"])))
        stock = stock
        stock_bd = self.spark.sparkContext.broadcast(
            stock.groupByKey().collectAsMap())
        rdd = wiki.flatMap(join_with_bd, preservesPartitioning=True)
        rdd = rdd.map(lambda x: {
            "triple": (x[1][0]["stock"], x[1][1]["topic"][0], x[1][1]["topic"][1]),
            "x2": x[1][0]["price"] ** 2,
            "xy_mean": x[1][0]["price"] * x[1][1]["mean"],
            "xy_max": x[1][0]["price"] * x[1][1]["max"],
            "xy_min": x[1][0]["price"] * x[1][1]["min"],
            "y2_mean": x[1][1]["mean"] ** 2,
            "y2_min": x[1][1]["min"] ** 2,
            "y2_max": x[1][1]["max"] ** 2
        })
        # no partition here, flatmap doesn't change partition
        rdd = rdd.keyBy(lambda x: x["triple"])
        rdd = rdd.reduceByKey(lambda x, y: {
            "triple": x["triple"],
            "x2": x["x2"] + y["x2"],
            "xy_mean": x["xy_mean"] + y["xy_mean"],
            "xy_max": x["xy_max"] + y["xy_max"],
            "xy_min": x["xy_min"] + y["xy_min"],
            "y2_mean": x["y2_mean"] + y["y2_mean"],
            "y2_min": x["y2_min"] + y["y2_min"],
            "y2_max": x["y2_max"] + y["y2_max"]
        })
        rdd = rdd.map(lambda x: {
            "triple": x[1]["triple"],
            "cos_mean": x[1]["xy_mean"] / (sqrt(x[1]["x2"]) * sqrt(x[1]["y2_mean"])),
            "cos_min": x[1]["xy_min"] / (sqrt(x[1]["x2"]) * sqrt(x[1]["y2_min"])),
            "cos_max": x[1]["xy_max"] / (sqrt(x[1]["x2"]) * sqrt(x[1]["y2_max"])),
        })
        self.cos_rdd = rdd

    def report_top_20(self):
        """report top 20 similarity combination
        """
        cos_rdd = self.cos_rdd.persist(storageLevel=StorageLevel.MEMORY_ONLY)
        for t in ("mean", "min", "max"):
            tmp = time.time()
            name = "cos_"+t
            l = cos_rdd.top(20, lambda x: x[name])
            rdd = self.spark.sparkContext.parallelize(l)
            rdd = rdd.map(lambda x: {
                "stock": x["triple"][0],
                "topic1": x["triple"][1],
                "topic2": x["triple"][2],
                "similarity": x[name]
            })
            rdd = self.report_lookup(rdd).sortBy(
                lambda x: x["similarity"], ascending=False)
            df: DataFrame = rdd.toDF()
            print("Top 20 Cosine Similarity Aggregation by "+t)
            df.show(truncate=False)
            print("Time consumed: ", time.time()-tmp)

    def start(self):
        """start calculation
        """
        self.title_index()
        self.date_index()
        self.calc_wiki_comb_bd()
        self.cosine_similarity_bd()
        self.report_top_20()


def spark_init() -> SparkSession:
    spark = SparkSession.builder\
        .config("spark.sql.shuffle.partitions", SPARK_PARTITION_NUM)\
        .config("spark.default.parallelism", SPARK_PARTITION_NUM)\
        .config("spark.rpc.message.maxSize", SPARK_FRAME_SIME)\
        .getOrCreate()
    return spark


SPARK_PARTITION_NUM = 12
SPARK_FRAME_SIME = 10
WIKI_PER_PARTITION = 2


def main():
    spark = spark_init()
    pre = Preparer(spark, path="../stock_sample.csv")
    df = pre.clean()
    df = pre.limit_stock_num(100)
    calc = Calculator(spark, df.rdd, wiki_path="wiki.csv")
    calc.start()


if __name__ == "__main__":
    main()
