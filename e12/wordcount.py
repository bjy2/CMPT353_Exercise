import sys
from pyspark.sql import SparkSession, functions
import string, re

spark = SparkSession.builder.appName('word count').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


def main(input_dir, output_dir):

    wordbreak = r'[%s\s]+' % re.escape(string.punctuation)

    lines_df = spark.read.text(input_dir)
    words_df = lines_df.select(functions.explode(functions.split(functions.lower(functions.col("value")), wordbreak)).alias("word"))
    words_df = words_df.filter(functions.col("word") != "")

    word_counts = words_df.groupBy("word").agg(functions.count("word").alias("count"))
    sorted_counts = word_counts.orderBy(functions.col("count").desc(), functions.col("word").asc())

    sorted_counts.write.csv(output_dir, header=False, mode="overwrite")
    spark.stop()

if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)