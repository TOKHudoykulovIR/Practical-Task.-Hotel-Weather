from pyspark.sql.functions import col, udf, coalesce
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession

from dotenv import load_dotenv

import pygeohash as pgh
import requests
import os

load_dotenv()  # load API key from .env
API_KEY = os.getenv("OPEN_CAGE_API_KEY")  # OpenCage API key

# create spark session
"""
spark.driver.memory - extra RAM to the driver program
spark.executor.memory - extra RAM to executors
"""
spark = (SparkSession.builder
         .appName("Task2")
         .config("spark.driver.memory", "8g")
         .config("spark.executor.memory", "8g")
         .config("spark.sql.shuffle.partitions", "200")
         .getOrCreate()
         )

data_folder = ".../restaurant_csv"  # path to restaurant_csv files
csv_files = [
    os.path.join(data_folder, f) for f in os.listdir(data_folder)  # os.listdir(data_folder) - return list of files by provided folder path
    if f.endswith(".csv")
]  # return list of paths of the csv files
df = spark.read.csv(csv_files, header=True, inferSchema=True)  # read CSV files

print(f"Total restaurant rows: {df.count()}")

def fetch_lat_lng(name):  # function to get missing lat and lnh
    if not name:
        return None, None
    try:
        response = requests.get(
            "https://api.opencagedata.com/geocode/v1/json",
            params={"q": name, "key": API_KEY, "limit": 1}, timeout=10
        ).json()
        if response["results"]:
            geometry = response["results"][0]["geometry"]
            return float(geometry["lat"]), float(geometry["lng"])
    except Exception as e:
        print(f"API error for {name}: {e}")
    return None, None


# get franchise_name with missed coordinates
missing_names = df.filter(col("lat").isNull() | col("lng").isNull()).select("franchise_name").distinct().collect()
print(f"Missing coordinates franchises: {missing_names}")

results = []
for row in missing_names:
    name = row["franchise_name"]
    lat, lng = fetch_lat_lng(name)  # call function to fill coordinates
    results.append((name, lat, lng))
print(f"Fetched coordinates data: {results}")

lookup_df = spark.createDataFrame(results, ["franchise_name", "new_lat", "new_lng"])  # create a spark df from python results

df = df.join(lookup_df, on="franchise_name", how="left")  # join the new coordinates to original df

# fill missing values using coalesce
df = (
    df.withColumn("lat", coalesce(col("lat"), col("new_lat")))
      .withColumn("lng", coalesce(col("lng"), col("new_lng")))
      .drop("new_lat", "new_lng")  # remove temporary helper columns
)

# takes lat and lng and returns a geohash string
geohash_udf = udf(
    lambda lat, lng: pgh.encode(lat, lng, precision=4),
    StringType()
)
df = df.withColumn("geohash", geohash_udf(col("lat"), col("lng")))  # apply the UDF to df
print(f"After adding geohash (example):")
df.show(n=1)

weather_root = ".../weather_data"  # path to weather folder
weather_df = spark.read.option("recursiveFileLookup", "true").parquet(weather_root)  # read everything inside

# to prevent column conflicts rename columns since both dfs have lat and lng in naming
weather_df = weather_df.withColumnRenamed("lat", "w_lat").withColumnRenamed("lng", "w_lng")
weather_df = weather_df.withColumn("geohash", geohash_udf(col("w_lat"), col("w_lng")))  # add geohash to weather data

enriched_df = df.join(weather_df, on="geohash", how="left")  # join df with weather_df
enriched_df = enriched_df.repartition("country", "city")  # repartition for better organized output folders in the next step

output_path = ".../enriched_data_parquet"  # path to save
enriched_df.write.mode("overwrite").partitionBy("country", "city").parquet(output_path)

print("Sample rows:")
enriched_df.show(5, truncate=False)

spark.stop()
