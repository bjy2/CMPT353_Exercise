import math
import sys
from pyspark.sql import SparkSession, functions, types, Row
import re

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        # TODO
        return Row(hostname = m.group(1), numbytes = m.group(2))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    rows = log_lines.map(line_to_row)
    rows = rows.filter(not_none)
    return rows


def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    # TODO: calculate r.
    count_requests = logs.groupby('hostname').agg(functions.count('hostname').alias("countRequests"))
    sum_requests_bytes = logs.groupby('hostname').agg(functions.sum('numbytes').alias("sumRequestsBytes"))

    aggregated_logs = count_requests.join(sum_requests_bytes, 'hostname')
    aggregated_logs = aggregated_logs.withColumns({
        "x": functions.col("countRequests"),
        "x^2": functions.col("countRequests") ** 2,
        "y": functions.col("sumRequestsBytes"),
        "y^2": functions.col("sumRequestsBytes") ** 2,
        "xy": functions.col("countRequests") * functions.col("sumRequestsBytes")
    })

    sums = aggregated_logs.agg(
        functions.sum('x').alias("x_sum"),
        functions.sum('x^2').alias("x2_sum"),
        functions.sum('y').alias("y_sum"),
        functions.sum('y^2').alias("y2_sum"),
        functions.sum('xy').alias("xy_sum")
    ).first()

    x_sum, x2_sum, y_sum, y2_sum, xy_sum = sums
    n = aggregated_logs.count()

    numerator = (n * xy_sum) - (x_sum * y_sum)
    denominator = (math.sqrt((n * x2_sum) - (x_sum ** 2))) * (math.sqrt((n * y2_sum) - (y_sum ** 2)))

    r = numerator / denominator if denominator != 0 else 0 # TODO: it isn't zero.

    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
