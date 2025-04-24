from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, from_json, current_date, current_timestamp, to_timestamp, create_map, lit, count
from pyspark.sql.types import StructType, StructField, StringType
import time

try:
    print("=================================================== Initializing Spark session ===================================================")

    # Initializing Spark Session
    spark = SparkSession.builder \
        .appName("KafkaConsumer") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5,io.delta:delta-spark_2.12:3.3.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

    print("=================================================== Spark session successfully initialized ===================================================")

    # Defining the schema for Kafka messages
    schema = StructType([
        StructField("Date/Time", StringType()),
        StructField("LV ActivePower (kW)", StringType()),
        StructField("Wind Speed (m/s)", StringType()),
        StructField("Theoretical_Power_Curve (KWh)", StringType()),
        StructField("Wind Direction (°)", StringType())
    ])

    # Reading data from Kafka in streaming fashion
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "winddata") \
        .option("startingOffsets", "earliest") \
        .load()

    print("=================================================== Successfully connected to Kafka topic ===================================================")

    # Printing raw Kafka data to the console
    raw_query = kafka_df.select(col("value").cast("string").alias("raw_data"))
    raw_query.writeStream \
        .format("console") \
        .outputMode("append") \
        .start()

    print("=================================================== Raw Kafka data is being printed to the console ===================================================")

    # Deserializing Kafka data and applying schema
    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).filter(col("data").isNotNull())

    print("=================================================== Kafka data successfully parsed and validated ===================================================")

    # Transforming the filtered data
    transformed_df = parsed_df.select(
        to_date(col("data.`Date/Time`"), "dd MM yyyy").alias("signal_date"),
        to_timestamp(col("data.`Date/Time`"), "dd MM yyyy HH:mm").alias("signal_ts"),
        current_date().alias("create_date"),
        current_timestamp().alias("create_ts"),
        create_map(
            lit("ActivePower_kW"), col("data.`LV ActivePower (kW)`"),
            lit("WindSpeed_m_s"), col("data.`Wind Speed (m/s)`"),
            lit("PowerCurve_kWh"), col("data.`Theoretical_Power_Curve (KWh)`"),
            lit("WindDirection_deg"), col("data.`Wind Direction (°)`")
        ).alias("signals")
    )

    print("=================================================== Data transformation applied ===================================================")

    # Counting the number of records in the transformed data
    def log_transformed_data_count(batch_df, batch_id):
        count = batch_df.count()
        print(f"Batch ID: {batch_id} | Transformed Data Count: {count}")

    transformed_query = transformed_df.writeStream \
        .foreachBatch(log_transformed_data_count) \
        .start()

    # Writing the transformed data to Delta table
    def log_delta_write(batch_df, batch_id):
        count = batch_df.count()
        print(f"Batch ID: {batch_id} | Writing {count} records to Delta table")
        batch_df.write.format("delta").mode("append").save("/home/xs535-himary/Downloads/Data-AI-main/data/delta/tables/wind_data")

    delta_stream = transformed_df.writeStream \
        .foreachBatch(log_delta_write) \
        .option("checkpointLocation", "/home/xs535-himary/Downloads/Data-AI-main/data/checkpoints/delta_kafka_consumer") \
        .start()

    print("=================================================== Delta table write initialized successfully ===================================================")

    # Automatic stop after a timeout (e.g., 60 seconds)
    timeout = 60  # Timeout in seconds
    start_time = time.time()

    while delta_stream.isActive:
        if time.time() - start_time > timeout:
            print("=================================================== Timeout reached, stopping the stream ===================================================")
            delta_stream.stop()
            break
        time.sleep(5)  # Check every 5 seconds

    # Stop the Spark session
    spark.stop()
    print("=================================================== Spark session stopped ===================================================")

except Exception as e:
    print(f"Critical failure: {e}")
    spark.stop()
    exit()