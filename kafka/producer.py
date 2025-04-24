from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import to_json, struct, col
import time

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("KafkaStreamingProducer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()

# Set log level to ERROR to suppress INFO and WARN logs
spark.sparkContext.setLogLevel("ERROR")

# Define the schema for the CSV file
schema = StructType([
    StructField("Date/Time", StringType(), True),
    StructField("LV ActivePower (kW)", DoubleType(), True),
    StructField("Wind Speed (m/s)", DoubleType(), True),
    StructField("Theoretical_Power_Curve (KWh)", DoubleType(), True),
    StructField("Wind Direction (°)", DoubleType(), True)
])

# Read the CSV file as a streaming source
df = spark.readStream.option("header", "true") \
    .schema(schema) \
    .csv("/home/xs535-himary/Downloads/Data-AI-main/data")  # Input directory for new CSV files

# Convert DataFrame to JSON format
json_df = df.select(to_json(struct(*[col(column) for column in df.columns])).alias("value"))

# Kafka Configurations
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "winddata"

# Write Data to Kafka Topic in streaming mode
try:
    record_count = 0  # Initialize record counter

    def update_record_count(batch_df, batch_id):
        global record_count
        batch_size = batch_df.count()
        record_count += batch_size
        print(f"Batch {batch_id}: Sent {batch_size} records. Total records sent: {record_count}")

    query = json_df.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("topic", KAFKA_TOPIC) \
        .outputMode("append") \
        .option("checkpointLocation", "../data/checkpoint") \
        .foreachBatch(update_record_count) \
        .start()
    
    print(f"================================Streaming data to Kafka topic '{KAFKA_TOPIC}'===============================================")
    time_limit_seconds = 9  # Set the desired time limit
    start_time = time.time()
    while query.isActive:
        if time.time() - start_time > time_limit_seconds:
            query.stop()
            print(f"================================Successfully published data to Kafka topic '{KAFKA_TOPIC}'============================================")
            print(f"Total records sent: {record_count}")
            break

except Exception as e:
    print(f"Error streaming data to Kafka: {e}")

# Stop Spark Session (This will execute when the streaming stops)
finally:
    print("[INFO] Stopping Spark session...")
    spark.stop()
    print("[INFO] Spark session stopped.")
