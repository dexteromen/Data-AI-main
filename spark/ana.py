from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, countDistinct, avg, hour, when
from pyspark.sql.types import StructType, StructField, StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("WindPowerAnalysis") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.3.0") \
    .getOrCreate()

# 1. Read data from Delta Lake
delta_table_path = "/home/xs535-himary/Downloads/Data-AI-main/data/delta/tables/wind_data"
df = spark.read.format("delta").load(delta_table_path)
print("================================================= Delta Table Loaded ==========================================================")
df.printSchema()  # Print schema to inspect signals column

# 2. Extract signal values from the signals column (assuming signals is a struct)
df = df.withColumn("LV ActivePower (kW)", col("signals.LV ActivePower (kW)")) \
       .withColumn("Wind Speed (m/s)", col("signals.Wind Speed (m/s)")) \
       .withColumn("Theoretical_Power_Curve (KWh)", col("signals.Theoretical_Power_Curve (KWh)")) \
       .withColumn("Wind Direction (°)", col("signals.Wind Direction (°)"))
print("Step 1: Signal values extracted from signals column.")

# 3. Calculate number of distinct signal_ts datapoints per day
daily_datapoints = df.groupBy(to_date("signal_ts").alias("date")).agg(countDistinct("signal_ts").alias("distinct_datapoints")) \
    .orderBy("date")
print("================================================= Daily Distinct Datapoints ==========================================================")
daily_datapoints.show()

# 4. Calculate average value of all signals per hour
hourly_averages = df.groupBy(to_date("signal_ts").alias("date"),hour("signal_ts").alias("hour")).agg(
        avg("LV ActivePower (kW)").alias("avg_active_power"),
        avg("Wind Speed (m/s)").alias("avg_wind_speed"),
        avg("Theoretical_Power_Curve (KWh)").alias("avg_theo_power"),
        avg("Wind Direction (°)").alias("avg_wind_direction")
    ) \
    .orderBy("date", "hour")
print("================================================= Hourly Averages ==========================================================")
hourly_averages.show()

# 5. Add generation_indicator column
df_with_indicator = df.withColumn("generation_indicator",
    when(col("LV ActivePower (kW)") < 200, "Low")
    .when((col("LV ActivePower (kW)") >= 200) & (col("LV ActivePower (kW)") < 600), "Medium")
    .when((col("LV ActivePower (kW)") >= 600) & (col("LV ActivePower (kW)") < 1000), "High")
    .otherwise("Exceptional")
)
print("Step 5: Generation indicator column added successfully.")

# 6. Create mapping DataFrame from JSON
mapping_json = [
    {"sig_name": "LV ActivePower (kW)", "sig_mapping_name": "active_power_average"},
    {"sig_name": "Wind Speed (m/s)", "sig_mapping_name": "wind_speed_average"},
    {"sig_name": "Theoretical_Power_Curve (KWh)", "sig_mapping_name": "theo_power_curve_average"},
    {"sig_name": "Wind Direction (°)", "sig_mapping_name": "wind_direction_average"}
]

schema = StructType([
    StructField("sig_name", StringType(), True),
    StructField("sig_mapping_name", StringType(), True)
])

mapping_df = spark.createDataFrame(mapping_json, schema)
print("Step 6: Mapping DataFrame created successfully.")

# 7. Rename columns based on mapping
mapping_dict = {row['sig_name']: row['sig_mapping_name'] for row in mapping_df.collect()}
df_final = df_with_indicator
for old_name, new_name in mapping_dict.items():
    if old_name in df_final.columns:
        df_final = df_final.withColumnRenamed(old_name, new_name)
print("================================================= Final DataFrame with Renamed Columns ==========================================================")
df_final.show()

# Clean up
spark.stop()