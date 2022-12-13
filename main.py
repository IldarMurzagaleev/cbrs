from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.functions import split, udf, desc, concat, col, lit
import pyspark.sql.functions as f
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
from pyspark.ml.linalg import SparseVector, VectorUDT, Vectors, _convert_to_vector, Vector
from pyspark.ml.feature import HashingTF, IDF, Tokenizer


spark = SparkSession.builder.appName('CBRS').getOrCreate()

def cbrs(items, all, k):
    """
    Function to calculate cosine similarity between request vector and table of features
    """
    # Apply similarity metric to the film_profile
    result = None
    for item in items:
        u = all.where(all.item_id==item)
        vm = u.collect()[0][1]
        target = u.collect()[0][0]
        dot_prod_udf = f.udf(lambda v: float(v.dot(vm) / (v.norm(2) * vm.norm(2))), FloatType())
        sim_df = all.withColumn('Similarity', dot_prod_udf('film_profile'))
        # Partition by item_id and order by the similarity in descending order
        window = Window.partitionBy(col("item_id")).orderBy((col("Similarity")).desc())
        # Add row numbers to the rows and get the top-k rows
        sim_df = sim_df.select(col('*'), row_number().over(window).alias('row_number')).where(col('Similarity') > 0.0)
        # Renaming
        sim_df = sim_df.withColumn("target_id", lit(target))
        cbrs_df = sim_df.select("target_id", col("item_id").alias("recommended_id"),  col("Similarity").alias("sim_rank")).limit(k)
        if result:
            result = result.union(cbrs_df)
        else:
            result = cbrs_df
    return result


table = spark.sparkContext.textFile("testx16x32_0.csv").flatMap(lambda line: line.split("\n"))
pairsCounts = table.map(lambda line: line.split(" ")).map(lambda line: (int(float(line[1])), line[0])).groupByKey().mapValues(' '.join)
df = spark.createDataFrame(pairsCounts).toDF("item_id", "user_id")

tokenizer = Tokenizer(inputCol="user_id", outputCol="users")
usersData = tokenizer.transform(df)
hashingTF = HashingTF(inputCol="users", outputCol="rawFeatures", numFeatures=1000000) #2197225
featurizedData = hashingTF.transform(usersData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
film_profile = rescaledData.select("item_id", col("features").alias("film_profile"))

# film_profile.rdd.saveAsTextFile("test")

k = 5
request = ["27544", "128790", "206578", "116292", "125530", "210830"]
#"114816", "32512", "40818",  "118884", "187844", "31734",
# request = request[:7]
user_rec = cbrs(request, film_profile , k)
user_rec.show(truncate=False)
user_rec.rdd.saveAsTextFile("out")

