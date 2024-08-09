# Databricks notebook source
# MAGIC %md
# MAGIC ## Importing the relevant libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, FMRegressor, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor, GeneralizedLinearRegression, IsotonicRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.functions import hour, month, year, day, minute, second, weekday, weekofyear, dayofweek, dayofmonth, dayofyear, col, corr, format_number, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# COMMAND ----------

spark = SparkSession.builder.appName("steel_energy_prediction").getOrCreate() # Initiate a Spark session

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the dataset

# COMMAND ----------

data = spark.read.csv("file:/Workspace/Users/n01606417@humber.ca/Steel_industry_data.csv",inferSchema=True,header=True)
data.printSchema()

# COMMAND ----------

data.count()

# COMMAND ----------

data.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis (EDA)

# COMMAND ----------

data.groupBy("Load_Type").count().show()

# COMMAND ----------

data.groupBy("Day_of_week").count().show()

# COMMAND ----------

data.groupBy("WeekStatus").count().show()

# COMMAND ----------

data.groupBy("Day_of_week").agg({"Usage_kWh": "avg"}).orderBy("avg(Usage_kWh)",ascending=False).withColumnRenamed("avg(Usage_kWh)","avg_energy_consumption").select(["Day_of_week",format_number("avg_energy_consumption",2).alias("avg_energy_consumption")]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Thursday has the highest average energy consumption while Sunday has the lowest average energy consumption.

# COMMAND ----------

data.groupBy("Load_Type").agg({"Usage_kWh": "avg"}).orderBy("avg(Usage_kWh)",ascending=False).withColumnRenamed("avg(Usage_kWh)","avg_energy_consumption").select(["Load_Type",format_number("avg_energy_consumption",2).alias("avg_energy_consumption")]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC As expected, maximum load has the highest energy consumption (in kWh) whereas light load has the least energy load.

# COMMAND ----------

data.groupBy("WeekStatus").agg({"Usage_kWh": "avg"}).orderBy("avg(Usage_kWh)",ascending=False).withColumnRenamed("avg(Usage_kWh)","avg_energy_consumption").select(["WeekStatus",format_number("avg_energy_consumption",2).alias("avg_energy_consumption")]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Average energy consumed is significantly higher on weekdays as compared to the weekends.

# COMMAND ----------

data.groupBy("Load_Type") \
    .agg({"NSM": "avg"}) \
    .orderBy("avg(NSM)", ascending=False) \
    .withColumnRenamed("avg(NSM)", "avg_nsm") \
    .select(
        "Load_Type",
        format_number("avg_nsm", 2).alias("avg_nsm")
    ) \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC Medium Load has the maximum NSM value while Light Load has the minimum NSM value.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization

# COMMAND ----------

data.createOrReplaceTempView("steel_energy") # Create a temporary view of the Spark dataframe

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from steel_energy;

# COMMAND ----------

# MAGIC %md
# MAGIC The target feature "Usage_kWh" has a highly right skewed distribution. In addition, the feature "CO2" has a right skewed distribution as well.

# COMMAND ----------

# MAGIC %md
# MAGIC NSM (Near-Surface Mounted) is considerably higher on weekdays in comparison to the weekends.

# COMMAND ----------

data.select(corr("CO2(tCO2)","Usage_kWh")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC There is a substantial positive correlation between C02 and energy consumption(in kWh).

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Rename the columns for better readability
cols_to_be_renamed = {
    "Lagging_Current_Reactive.Power_kVarh": "Lagging_Current_Reactive_Power_kVarh",
    "CO2(tCO2)": "CO2"
}

for key, val in cols_to_be_renamed.items():
    data = data.withColumnRenamed(key,val)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Splitting the dataset into train and test sets

# COMMAND ----------

train, test = data.randomSplit([0.75,0.25],seed=64)

# COMMAND ----------

train.count(), test.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating an end-to-end feature engineering and machine learning pipeline

# COMMAND ----------

day_of_week_indexer = StringIndexer(inputCol='Day_of_week',outputCol='Day_of_week_index')
load_type_indexer = StringIndexer(inputCol='Load_Type',outputCol='Load_Type_index')
week_status_indexer = StringIndexer(inputCol='WeekStatus',outputCol='WeekStatus_index')
assembler = VectorAssembler(inputCols=["Lagging_Current_Reactive_Power_kVarh","Leading_Current_Reactive_Power_kVarh","CO2",
"Lagging_Current_Power_Factor","Leading_Current_Power_Factor","NSM","Day_of_week_index","Load_Type_index","WeekStatus_index"], outputCol="features", handleInvalid="skip")
scaler = StandardScaler(inputCol="features",outputCol="scaledFeatures")
lr = LinearRegression(featuresCol="scaledFeatures",labelCol="Usage_kWh")

# COMMAND ----------

pipeline = Pipeline(stages=[
    day_of_week_indexer,load_type_indexer,week_status_indexer,assembler,scaler,lr
])
pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training & Evaluation

# COMMAND ----------

fitted_pipelines = []
model_names = []
r2_scores = []
rmse_scores = []
mae_scores = []
mse_scores = []
explained_variance_scores = []

# COMMAND ----------

def data_prep_ml_pipeline(model: Pipeline) -> None:
    model_names.append(str(model).split('(')[0])
    day_of_week_indexer = StringIndexer(inputCol='Day_of_week',outputCol='Day_of_week_index')
    load_type_indexer = StringIndexer(inputCol='Load_Type',outputCol='Load_Type_index')
    week_status_indexer = StringIndexer(inputCol='WeekStatus',outputCol='WeekStatus_index')
    assembler = VectorAssembler(inputCols=["Lagging_Current_Reactive_Power_kVarh","Leading_Current_Reactive_Power_kVarh","CO2",
    "Lagging_Current_Power_Factor","Leading_Current_Power_Factor","NSM","Day_of_week_index","Load_Type_index","WeekStatus_index"], outputCol="features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features",outputCol="scaledFeatures")
    pipeline = Pipeline(stages=[
        day_of_week_indexer,load_type_indexer,week_status_indexer,assembler,scaler,model
    ])
    model = pipeline.fit(train)
    predictions = model.transform(test)
    r2_eval = RegressionEvaluator(labelCol="Usage_kWh", predictionCol="prediction", metricName="r2")
    rmse_eval = RegressionEvaluator(labelCol="Usage_kWh", predictionCol="prediction", metricName="rmse")
    mae_eval = RegressionEvaluator(labelCol="Usage_kWh", predictionCol="prediction", metricName="mae")
    mse_eval = RegressionEvaluator(labelCol="Usage_kWh", predictionCol="prediction", metricName="mse")
    explained_var_eval = RegressionEvaluator(labelCol="Usage_kWh", predictionCol="prediction", metricName="var")
    r2 = r2_eval.evaluate(predictions)
    rmse = rmse_eval.evaluate(predictions)
    mae = mae_eval.evaluate(predictions)
    mse = mse_eval.evaluate(predictions)
    explained_var = explained_var_eval.evaluate(predictions)
    print("R2: %f" % r2)
    print("RMSE: %f" % rmse)
    print("MAE: %f" % mae)
    print("MSE: %f" % mse)
    print("Explained Variance: %f" % explained_var)
    fitted_pipelines.append(pipeline)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mse_scores.append(mse)
    explained_variance_scores.append(explained_var)

# COMMAND ----------

data_prep_ml_pipeline(LinearRegression(labelCol='Usage_kWh',featuresCol='scaledFeatures'))

# COMMAND ----------

data_prep_ml_pipeline(DecisionTreeRegressor(labelCol='Usage_kWh',featuresCol='scaledFeatures'))

# COMMAND ----------

data_prep_ml_pipeline(RandomForestRegressor(labelCol='Usage_kWh',featuresCol='scaledFeatures'))

# COMMAND ----------

data_prep_ml_pipeline(GBTRegressor(labelCol='Usage_kWh',featuresCol='scaledFeatures',predictionCol='prediction'))

# COMMAND ----------

data_prep_ml_pipeline(FMRegressor(labelCol='Usage_kWh',featuresCol='scaledFeatures'))

# COMMAND ----------

data_prep_ml_pipeline(GeneralizedLinearRegression(family='poisson',labelCol='Usage_kWh',featuresCol='scaledFeatures'))

# COMMAND ----------

data_prep_ml_pipeline(GeneralizedLinearRegression(family='tweedie',labelCol='Usage_kWh',featuresCol='scaledFeatures'))

# COMMAND ----------

data_prep_ml_pipeline(IsotonicRegression(labelCol='Usage_kWh',featuresCol='scaledFeatures'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Models Performance Comparison

# COMMAND ----------

schema = StructType([
    StructField("Model", StringType(), True),
    StructField("R2", DoubleType(), True),
    StructField("MAE", DoubleType(), True),
    StructField("MSE", DoubleType(), True),
    StructField("RMSE", DoubleType(), True),
    StructField("Explained Variance", DoubleType(), True)
])

model_perfs = spark.createDataFrame([
    {'Model': model_names[i],
     'Pipeline': fitted_pipelines[i],
     'R2': r2_scores[i],
     'MAE': mae_scores[i],
     'MSE': mse_scores[i],
     'RMSE': rmse_scores[i],
     'Explained Variance': explained_variance_scores[i]}
    for i in range(len(fitted_pipelines))
], schema).orderBy('R2', ascending=False)

display(model_perfs)

# COMMAND ----------

# MAGIC %md
# MAGIC The GBT regressor model has emerged as the best performing model having achieved a remarkable r2 score of more than 99% on the test set.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning & Cross Validation

# COMMAND ----------

def tune_hyperparameters(model,param_grid):
    day_of_week_indexer = StringIndexer(inputCol='Day_of_week',outputCol='Day_of_week_index')
    load_type_indexer = StringIndexer(inputCol='Load_Type',outputCol='Load_Type_index')
    week_status_indexer = StringIndexer(inputCol='WeekStatus',outputCol='WeekStatus_index')
    assembler = VectorAssembler(inputCols=["Lagging_Current_Reactive_Power_kVarh","Leading_Current_Reactive_Power_kVarh","CO2",
    "Lagging_Current_Power_Factor","Leading_Current_Power_Factor","NSM","Day_of_week_index","Load_Type_index","WeekStatus_index"], outputCol="features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features",outputCol="scaledFeatures")
    pipeline = Pipeline(stages=[
        day_of_week_indexer,load_type_indexer,week_status_indexer,assembler,scaler,model
    ])
    train_val_split = TrainValidationSplit(estimator=pipeline,evaluator=RegressionEvaluator().setMetricName("r2").setLabelCol("Usage_kWh"),estimatorParamMaps=param_grid,trainRatio=0.8)
    optimized_model = train_val_split.fit(train)
    predictions = optimized_model.transform(test)
    r2 = RegressionEvaluator(metricName="r2",labelCol="Usage_kWh").evaluate(predictions)
    print("R2 score on test data = %g" % r2)
    rmse = RegressionEvaluator(metricName="rmse",labelCol="Usage_kWh").evaluate(predictions)
    print("RMSE on test data = %g" % rmse)
    mae = RegressionEvaluator(metricName="mae",labelCol="Usage_kWh").evaluate(predictions)
    print("MAE on test data = %g" % mae)
    mse = RegressionEvaluator(metricName="mse",labelCol="Usage_kWh").evaluate(predictions)
    print("MSE on test data = %g" % mse)
    explained_variance = RegressionEvaluator(metricName="var",labelCol="Usage_kWh").evaluate(predictions)
    print("Explained Variance on test data = %g" % explained_variance)
    fitted_pipelines.append(pipeline)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mse_scores.append(mse)
    explained_variance_scores.append(explained_variance)
    model_names.append(str(model).split("(")[0])

# COMMAND ----------

lr_param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.fitIntercept, [True, False]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

tune_hyperparameters(LinearRegression(labelCol='Usage_kWh',featuresCol='scaledFeatures'),lr_param_grid)

# COMMAND ----------

dt = DecisionTreeRegressor(labelCol="Usage_kWh",featuresCol="scaledFeatures")
dt_param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [2, 5, 10]) \
    .addGrid(dt.maxBins, [32, 64, 128]) \
    .build()

tune_hyperparameters(dt,dt_param_grid)

# COMMAND ----------

rf = RandomForestRegressor(labelCol="Usage_kWh",featuresCol="scaledFeatures")
rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 50]) \
    .addGrid(rf.featureSubsetStrategy, ['all','auto','onethird','sqrt','log2']) \
    .build()

tune_hyperparameters(rf,rf_param_grid)

# COMMAND ----------

fm = FMRegressor(labelCol="Usage_kWh",featuresCol="scaledFeatures")
fm_param_grid = ParamGridBuilder() \
    .addGrid(fm.stepSize, [0.001, 0.01, 0.1]) \
    .addGrid(fm.factorSize, [4, 8, 16]) \
    .build()

tune_hyperparameters(fm,fm_param_grid)

# COMMAND ----------

iso = IsotonicRegression(labelCol="Usage_kWh",featuresCol="scaledFeatures")
iso_param_grid = ParamGridBuilder() \
    .addGrid(iso.isotonic, [True, False]) \
    .build()

tune_hyperparameters(iso,iso_param_grid)

# COMMAND ----------

poisson = GeneralizedLinearRegression(family="poisson",labelCol="Usage_kWh",featuresCol="scaledFeatures")
poisson_param_grid = ParamGridBuilder() \
    .addGrid(poisson.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(poisson.fitIntercept, [True, False]) \
    .build()

tune_hyperparameters(poisson,poisson_param_grid)

# COMMAND ----------

tweedie = GeneralizedLinearRegression(family="tweedie",labelCol="Usage_kWh",featuresCol="scaledFeatures")
tweedie_param_grid = ParamGridBuilder() \
    .addGrid(tweedie.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(tweedie.fitIntercept, [True, False]) \
    .addGrid(tweedie.variancePower, [0.0, 1.0]) \
    .build()

tune_hyperparameters(tweedie,tweedie_param_grid)

# COMMAND ----------

gbt = GBTRegressor(labelCol="Usage_kWh",featuresCol="scaledFeatures")
gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 10]) \
    .addGrid(gbt.maxIter, [10, 20]) \
    .build()

tune_hyperparameters(gbt,gbt_param_grid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimized Models Performance Comparison

# COMMAND ----------

schema = StructType([
    StructField("Model", StringType(), True),
    StructField("R2", DoubleType(), True),
    StructField("MAE", DoubleType(), True),
    StructField("MSE", DoubleType(), True),
    StructField("RMSE", DoubleType(), True),
    StructField("Explained Variance", DoubleType(), True)
])

model_perfs = spark.createDataFrame([
    {'Model': model_names[i],
     'Pipeline': fitted_pipelines[i],
     'R2': r2_scores[i],
     'MAE': mae_scores[i],
     'MSE': mse_scores[i],
     'RMSE': rmse_scores[i],
     'Explained Variance': explained_variance_scores[i]}
    for i in range(len(fitted_pipelines))
], schema).orderBy('R2', ascending=False)

display(model_perfs)

# COMMAND ----------

# MAGIC %md
# MAGIC So, finally, after completing hyperparameter tuning, the Decision Tree Regressor has turned out to be the best performing model as it has achieved an incredible r2 score of almost 99.5% on the test set.

# COMMAND ----------

bestModel, bestModelPipeline = model_names[9], fitted_pipelines[9]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross validating the performance of the best performing model

# COMMAND ----------

dt_evaluator = RegressionEvaluator(labelCol="Usage_kWh",metricName="r2")

dt_cv = CrossValidator(estimator=bestModelPipeline,
                       evaluator=dt_evaluator,
                       estimatorParamMaps=dt_param_grid,
                       numFolds=3)

dt_cv_model = dt_cv.fit(train)
predictions = dt_cv_model.transform(test)
print("Cross validation R2 score:", dt_evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving the best performing model for deployment

# COMMAND ----------

bestModelPipeline.save("file:/Workspace/Users/n01606417@humber.ca/steel_energy_prediction_pipeline")