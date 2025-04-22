from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns


# Initialize Spark Session
spark = SparkSession.builder \
    .appName("WindPowerMLTask") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.3.0") \
    .getOrCreate()

# Load the processed data
processed_data_path = "/home/xs535-himary/Downloads/Data-AI-main/data/processed/renamed_data" 
renamed_df = spark.read.format("delta").load(processed_data_path)
print("\n================================================= Processed Data Loaded ==========================================================")
renamed_df.show(truncate=False)

# Encode `generation_status` column as numeric using StringIndexer
indexer = StringIndexer(inputCol="generation_status", outputCol="generation_status_index")
indexed_df = indexer.fit(renamed_df).transform(renamed_df)
print("\n================================================= Indexed Data ==========================================================")
indexed_df.show(truncate=False)

# Combine features into a single vector column using VectorAssembler
feature_columns = ["wind_speed_average", "theo_power_curve_average", "wind_direction_average", "generation_status_index"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
final_df = assembler.transform(indexed_df)
print("\n================================================= final Data ==========================================================")
final_df.show(truncate=False)

# Split data into training and testing 
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)



# # Train a Decision Tree Regressor to predict active power
# dt = DecisionTreeRegressor(featuresCol="features", labelCol="active_power_average")
# dt_model = dt.fit(train_df)

# single_day_df = final_df.filter(col("signal_date") == "2018-01-01")
# # Predict for a single day's data
# single_day_predictions = dt_model.transform(single_day_df)
# print("\n====================================================== Predictions for Single Day ========================================================")
# single_day_predictions.select("signal_date", "hour", "active_power_average", "prediction").show(truncate=False)

# # Evaluate the model
# predictions = dt_model.transform(test_df)
# evaluator = RegressionEvaluator(labelCol="active_power_average", predictionCol="prediction", metricName="rmse")
# rmse = evaluator.evaluate(predictions)
# print(f"\nRoot Mean Square Error (RMSE): {rmse}")

# evaluator = RegressionEvaluator(labelCol="active_power_average", predictionCol="prediction", metricName="mae")
# mae = evaluator.evaluate(predictions)
# print(f"\nMean Absolute Error (MAE): {mae}")

# # === Calculate R-Squared ===
# evaluator_r2 = RegressionEvaluator(labelCol="active_power_average", predictionCol="prediction", metricName="r2")
# r_squared = evaluator_r2.evaluate(predictions)
# print(f"\nR-Squared: {r_squared:.2f}")

# === Convert predictions to Pandas DataFrame for visualization (same as before) ===
# pandas_df = predictions.select("active_power_average", "prediction", "wind_speed_average", "theo_power_curve_average", "wind_direction_average").toPandas()




# Train a Linear Regression model to predict active power
lr = LinearRegression(featuresCol="features", labelCol="active_power_average")
lr_model = lr.fit(train_df)


# Predict for a single day's data
single_day_df = final_df.filter(col("signal_date") == "2018-01-19")
single_day_predictions = lr_model.transform(single_day_df)
print("\n====================================================== Predictions for Single Day ========================================================")
single_day_predictions.select("signal_date", "hour", "active_power_average", "prediction").show(truncate=False)

# Evaluate the model
predictions = lr_model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="active_power_average", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"\nRoot Mean Square Error (RMSE): {rmse}")

evaluator = RegressionEvaluator(labelCol="active_power_average", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)
print(f"\nMean Absolute Error (MAE): {mae}")

# === Calculate R-Squared ===
evaluator_r2 = RegressionEvaluator(labelCol="active_power_average", predictionCol="prediction", metricName="r2")
r_squared = evaluator_r2.evaluate(predictions)
print(f"\nR-Squared: {r_squared:.2f}")

# === Calculate MAE ===
evaluator_mae = RegressionEvaluator(labelCol="active_power_average", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)
print(f"\nMean Absolute Error (MAE): {mae:.2f}")

# Convert predictions to Pandas DataFrame for visualization
pandas_df = predictions.select("active_power_average", "prediction", "wind_speed_average", "theo_power_curve_average", "wind_direction_average").toPandas()

# === Graph 1: Actual vs Predicted Active Power ===
plt.figure(figsize=(10, 6))
plt.scatter(pandas_df["active_power_average"], pandas_df["prediction"], alpha=0.5)
plt.xlabel("Actual Active Power")
plt.ylabel("Predicted Active Power")
plt.title("Actual vs Predicted Active Power")
plt.plot([0, max(pandas_df["active_power_average"])], [0, max(pandas_df["active_power_average"])], color='red', linestyle='--')
plt.savefig("Actual_vs_Predicted.png")

# === Graph 2: Error Distribution (Residuals) ===
pandas_df["residuals"] = pandas_df["active_power_average"] - pandas_df["prediction"]
plt.figure(figsize=(10, 6))
sns.histplot(pandas_df["residuals"], kde=True, color='red')
plt.xlabel("Residuals")
plt.title("Error Distribution (Residuals)")
plt.axvline(0, color='black', linestyle='--')
plt.savefig("Error_Distribution.png")


