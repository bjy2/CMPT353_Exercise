import sys
from pyspark.sql import SparkSession, functions, types
import re

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

wiki_schema = types.StructType([
types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('views', types.IntegerType()),
    types.StructField('bytes', types.LongType()),
])

def extract_day_hour(path):
    match = re.search(r'pagecounts-(\d{8}-\d{2})', path)
    return match.group(1) if match else None


def main(in_directory, out_directory):
    data = spark.read.csv(in_directory, schema=wiki_schema, sep=' ').withColumn('filename', functions.input_file_name())
    path_to_hour = functions.udf(extract_day_hour, returnType=types.StringType())
    data = data.withColumn('hour', path_to_hour(data['filename']))
    data = data.filter(data['Language'] == 'en')
    data = data.filter(data['Title'] != 'Main_Page')
    filtered_data = data.filter(data['title'].startswith('Special:') == False)

    filtered_data = filtered_data.cache()

    max_views_per_hour = filtered_data.groupBy('hour').agg(functions.max('views').alias('max_views'))
    most_viewed = filtered_data.join(max_views_per_hour, ['hour'], 'inner').filter(filtered_data.views == max_views_per_hour.max_views)
    result = most_viewed.select('hour', 'title', 'views').orderBy('hour', 'title')
    result.write.csv(out_directory + '-wikipedia', mode='overwrite')

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)