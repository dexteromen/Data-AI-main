from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark Session with Delta Lake support
spark = SparkSession.builder \
    .appName("WindPowerMLTask") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.3.0") \
    .getOrCreate()

# Load the processed data from Delta Lake
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
print("\n================================================= Final Data ==========================================================")
final_df.show(truncate=False)

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = final_df.select("wind_speed_average", "theo_power_curve_average", "wind_direction_average", 
                            "generation_status_index", "active_power_average").toPandas()

# Define features (X) and target (y)
X = pandas_df[["wind_speed_average", "theo_power_curve_average", "wind_direction_average", "generation_status_index"]]
y = pandas_df["active_power_average"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model using scikit-learn
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = mse ** 0.5  # Root Mean Square Error
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
r2 = r2_score(y_test, y_pred)  # R-Squared

print(f"\nRoot Mean Square Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-Squared: {r2:.2f}")

# Visualizations
# === Graph 1: Actual vs Predicted Active Power ===
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Active Power")
plt.ylabel("Predicted Active Power")
plt.title("Actual vs Predicted Active Power")
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.savefig("Actual_vs_Predicted_sklearn.png")

# === Graph 2: Error Distribution (Residuals) ===
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='red')
plt.xlabel("Residuals")
plt.title("Error Distribution (Residuals)")
plt.axvline(0, color='black', linestyle='--')
plt.savefig("Error_Distribution_sklearn.png")