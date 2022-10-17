from datetime import date, timedelta
from pyspark import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, FloatType, IntegerType, DateType
from pyspark import StorageLevel
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import sys
import os


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


def spark_init() -> SparkSession:
    spark = SparkSession.builder\
        .config("spark.sql.shuffle.partitions", SPARK_PARTITION_NUM)\
        .config("spark.default.parallelism", SPARK_PARTITION_NUM)\
        .config("spark.rpc.message.maxSize", SPARK_FRAME_SIME)\
        .getOrCreate()
    return spark


SPARK_PARTITION_NUM = 12
SPARK_FRAME_SIME = 10


def main():
    spark = spark_init()
    pre = Preparer(spark, "../stock_sample.csv", print_debug=True)
    df = pre.clean()
    df = pre.limit_stock_num(1000)
    df.collect()


if __name__ == "__main__":
    main()
