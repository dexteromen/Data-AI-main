from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, hour, avg, col, when, broadcast

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("WindPowerAnalysis") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.3.0") \
    .getOrCreate()

# Load the Delta Table
delta_table_path = "/home/xs535-himary/Downloads/Data-AI-main/data/delta/tables/wind_data"
delta_df = spark.read.format("delta").load(delta_table_path)
print("\n================================ Query 1: SCHEMA ===========================================")
delta_df.printSchema()  # Print schema to inspect signals column

# === Query 1: Distinct Count of signal_ts Per Day ===
distinct_count_df = delta_df.groupBy("signal_date").agg(countDistinct("signal_ts").alias("distinct_signal_count"))
print("\n================================ Query 2: Distinct Count of signal_ts Per Day ===========================================")
distinct_count_df.show(truncate=False)

# === Query 2: Average Values for All Signals Per Hour ===
averages_per_hour = delta_df.groupBy("signal_date", hour("signal_ts").alias("hour")).agg(
    avg(col("signals").getItem("ActivePower_kW").cast("float")).alias("avg_ActivePower_kW"),
    avg(col("signals").getItem("WindSpeed_m_s").cast("float")).alias("avg_WindSpeed_m_s"),
    avg(col("signals").getItem("PowerCurve_kWh").cast("float")).alias("avg_PowerCurve_kWh"),
    avg(col("signals").getItem("WindDirection_deg").cast("float")).alias("avg_WindDirection_deg")
)
print("\n================================ Query 3: Average Values for All Signals Per Hour ===========================================")
averages_per_hour.show(truncate=False)

# === Query 3: Adding Generation Indicator Based on Average ActivePower ===
averaged_with_indicator = averages_per_hour.withColumn(
    "generation_indicator",
    when(col("avg_ActivePower_kW") < 200, "Low")
    .when((col("avg_ActivePower_kW") >= 200) & (col("avg_ActivePower_kW") < 600), "Medium")
    .when((col("avg_ActivePower_kW") >= 600) & (col("avg_ActivePower_kW") < 1000), "High")
    .otherwise("Exceptional")
)
print("\n================================ Query 4: Adding Generation Indicator to Averages ===========================================")
averaged_with_indicator.show(truncate=False)

# === Step 4: Create Signal Name Mapping Using Spark SQL ===
spark.sql("""
    CREATE OR REPLACE TEMP VIEW mapping_table AS
    SELECT 'avg_ActivePower_kW' AS sig_name, 'active_power_average' AS sig_mapping_name
    UNION ALL
    SELECT 'avg_WindSpeed_m_s', 'wind_speed_average'
    UNION ALL
    SELECT 'avg_PowerCurve_kWh', 'theo_power_curve_average'
    UNION ALL
    SELECT 'avg_WindDirection_deg', 'wind_direction_average'
    UNION ALL
    SELECT 'generation_indicator', 'generation_status'
""")

mapping_df = spark.table("mapping_table")
print("\n================================ Query 5: Mapping DataFrame Created Using Spark SQL ===========================================")
mapping_df.show(truncate=False)

# === Step 5: Perform Renaming Columns Using Broadcast Join ===
# Explode the mapping into a DataFrame and broadcast it
renamed_df_with_broadcast = averaged_with_indicator

for row in broadcast(mapping_df).collect():
    sig_name = row["sig_name"]
    sig_mapping_name = row["sig_mapping_name"]
    if sig_name in renamed_df_with_broadcast.columns:
        renamed_df_with_broadcast = renamed_df_with_broadcast.withColumnRenamed(sig_name, sig_mapping_name)

print("\n================================ Query 6: Final DataFrame with Renamed Signal Names ===========================================")
renamed_df_with_broadcast.show(truncate=False)

# Save the processed DataFrame for the ML task
processed_data_path = "/home/xs535-himary/Downloads/Data-AI-main/data/processed/renamed_data"
renamed_df_with_broadcast.write.mode("overwrite").format("delta").save(processed_data_path)
print("\nProcessed data saved to:", processed_data_path)

print("\n================================ Completed ===========================================")