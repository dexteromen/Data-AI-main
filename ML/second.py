from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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

# === Model 1: Linear Regression ===
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluate Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mse_lr ** 0.5
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n=== Linear Regression ===")
print(f"Root Mean Square Error (RMSE): {rmse_lr}")
print(f"Mean Absolute Error (MAE): {mae_lr}")
print(f"R-Squared: {r2_lr:.2f}")

# === Model 2: Decision Tree Regressor ===
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Evaluate Decision Tree Regressor
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = mse_dt ** 0.5
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\n=== Decision Tree Regressor ===")
print(f"Root Mean Square Error (RMSE): {rmse_dt}")
print(f"Mean Absolute Error (MAE): {mae_dt}")
print(f"R-Squared: {r2_dt:.2f}")

# Visualizations for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, label="Linear Regression")
plt.scatter(y_test, y_pred_dt, alpha=0.5, label="Decision Tree", color='orange')
plt.xlabel("Actual Active Power")
plt.ylabel("Predicted Active Power")
plt.title("Actual vs Predicted Active Power")
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label="Ideal Fit")
plt.legend()
plt.savefig("Actual_vs_Predicted_Comparison.png")

# Error Distribution for Linear Regression
residuals_lr = y_test - y_pred_lr
residuals_dt = y_test - y_pred_dt

plt.figure(figsize=(10, 6))
sns.histplot(residuals_lr, kde=True, color='blue', label="Linear Regression")
sns.histplot(residuals_dt, kde=True, color='orange', label="Decision Tree")
plt.xlabel("Residuals")
plt.title("Error Distribution (Residuals)")
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.savefig("Error_Distribution_Comparison.png")