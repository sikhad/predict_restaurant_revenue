import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.functions import year
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import QuantileDiscretizer
from itertools import chain
from numpy import array
import numpy as np
import pandas as pd
import datetime

class PredictRevenue:

	def __init__(self):

		# set up spark session
		spark = SparkSession.builder.appName('SalesRev').getOrCreate()
		sc = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

		self.spark = spark
		self.sc = sc
	
	# read data to be trained on
	def read_data(self):

		df = None

		schema = StructType([
			StructField("Store_ID", StringType()),
			StructField("Fiscal_Qtr", IntegerType()),
			StructField("DateStringYYYYMMDD", StringType()),
			StructField("Fiscal_dayofWk", IntegerType()),
			StructField("Daypart", StringType()),
			StructField("HourlyWeather", StringType()),
			StructField("Hour", IntegerType()),
			StructField("AvgHourlyTemp", DoubleType()),
			StructField("SalesRevenue", DoubleType())])

		df = self.sc.read.format("com.databricks.spark.csv")\
				.schema(schema)\
				.option("header", "true")\
				.option("mode", "DROPMALFORMED")\
				.load("SalesbyHour.csv")

		return df

	# read data to be predicted on
	def read_new_data(self):

		df = None

		schema = StructType([
			StructField("Store_ID", StringType()),
			StructField("Fiscal_Qtr", IntegerType()),
			StructField("DateStringYYYYMMDD", StringType()),
			StructField("Fiscal_dayofWk", IntegerType()),
			StructField("Daypart", StringType()),
			StructField("HourlyWeather", StringType()),
			StructField("Hour", IntegerType()),
			StructField("AvgHourlyTemp", DoubleType())])

		df = self.sc.read.format("com.databricks.spark.csv")\
				.schema(schema)\
				.option("header", "true")\
				.option("mode", "DROPMALFORMED")\
				.load("SalesbyHour_predict.csv")

		return df

	# transform data based on different conditions
	def data_transformation(self, df):

		df = df.withColumn("date", from_unixtime(unix_timestamp('DateStringYYYYMMDD', 'yyyyMMdd')))
		df = df.withColumn("Year", year("date")) # extract year
		df = df.withColumn("Day", dayofmonth("date")) # extract day
		df = df.withColumn("Month", month("date")) # extract month

		# bin continuous variable based on quantiles (temperature)
		discretizer = QuantileDiscretizer(numBuckets=6, inputCol="AvgHourlyTemp", outputCol="temp_quantile")
		result = discretizer.fit(df).transform(df)
		df = result

		# based on variation seen in boxplots
		df = df.withColumn(
			'store_variability',
			F.when((F.col("Store_ID").isin('16', '17', '18', '20', '22', '31')), "variable")\
			.when((F.col("Store_ID").isin('11', '2', '21', '23', '32', '34', '36', '38')), "not-variable")
			)

		# based on mean revenue per store id
		df = df.withColumn(
			'store_rank',
			F.when((F.col("Store_ID").isin('31', '17', '20', '38', '2', '36', '21')), "great")\
			.when((F.col("Store_ID").isin('18', '32', '34', '23')), "good")\
			.when((F.col("Store_ID").isin('11', '22', '16')), "ok")
			)

		return df

	# build linear regression model
	def build_linreg(self, trainingData, testData):
	
		lr = LinearRegression(labelCol="SalesRevenue", featuresCol="features", 
							  maxIter=100, regParam=1, elasticNetParam=0.0)

		lrModel = lr.fit(trainingData)
		trainingSummary = lrModel.summary

		# calculate summary, rmse for train
		print("numIterations: %d" % trainingSummary.totalIterations)
		print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
		print("Train RMSE: %f" % trainingSummary.rootMeanSquaredError)
		print("Train r2: %f" % trainingSummary.r2)

		# calculate summary, rmse for test
		lrModel_test = lr.fit(testData)
		testingSummary = lrModel_test.summary
		print("Test RMSE: %f" % testingSummary.rootMeanSquaredError)
		print("Test r2: %f" % testingSummary.r2)

		return lrModel

	# build gbt model
	def build_gbt(self, trainingData, testData):

		gbt = GBTRegressor(featuresCol = 'features', labelCol = 'SalesRevenue', maxIter=100, maxDepth=5)
		
		gbt_model = gbt.fit(trainingData)
		
		# calculate rmse
		gbt_evaluator = RegressionEvaluator(labelCol="SalesRevenue", predictionCol="prediction", metricName="rmse")
		
		gbt_predictions_train = gbt_model.transform(trainingData)
		rmse = gbt_evaluator.evaluate(gbt_predictions_train)
		print("Train RMSE = %g" % rmse)

		gbt_predictions = gbt_model.transform(testData)
		rmse = gbt_evaluator.evaluate(gbt_predictions)
		print("Test RMSE = %g" % rmse)

		return gbt_model

	# predict based on linear regression model
	def fit_linreg(self, lrModel, newData):

		lrmodel_predictions = None

		lrmodel_predictions = lrModel.transform(newData)

		return lrmodel_predictions

	# predict based on gbt model
	def fit_gbt(self, gbt_model, newData):

		gbt_predictions = None

		gbt_predictions = gbt_model.transform(newData)

		return gbt_predictions

	# encode categorical features and vectorize all features for the pipeline
	def encode_vectorize(self, categoricalColumns, numericCols):
		
		stages = []

		for categoricalCol in categoricalColumns:
			stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
			encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
			stages += [stringIndexer, encoder]

		assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
		assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
		stages += [assembler]

		return stages

	# remove outliers from data
	def remove_outliers(self, dataset):

		dataset_trunc = None

		dataset_trunc = dataset.where("SalesRevenue < 1000 and SalesRevenue > -400")

		return dataset_trunc

	# define the pipeline for training
	def create_pipeline_train(self, stages, df, cols):

		dataset = None

		pipeline = Pipeline(stages=stages)
		pipelineModel = pipeline.fit(df)
		dataset = pipelineModel.transform(df)

		selectedcols = ["features"] + cols
		dataset = dataset.select(selectedcols)

		return dataset

	# define the pipeline for prediction
	def create_pipeline_predict(self, stages, df, df_new, cols):

		dataset = None

		pipeline = Pipeline(stages=stages)
		pipelineModel = pipeline.fit(df) # fit the training dataset
		dataset = pipelineModel.transform(df_new) # transform on new

		selectedcols = ["features"] + cols
		dataset = dataset.select(selectedcols)

		return dataset

	# split data into train and test, using this break gave about 80/20 split 
	def split_data(self, dataset):
	
	    trainingData = dataset.where("Day < 24")
	    testData = dataset.where("Day > 24")

	    print(trainingData.count())
	    print(testData.count())
		
	    return trainingData, testData

def main():

	# initiate class
	predictRev = PredictRevenue()

	# pick the columns for the model
	categoricalColumns = ['Store_ID', 'Fiscal_dayofWk', 'HourlyWeather', 'Hour', 'Daypart', 'Year', 'Month', 
					 'store_variability', 'store_rank', 'temp_quantile']
	numericCols = []

	# read and transform data
	df = predictRev.read_data()
	df = predictRev.data_transformation(df)

	# build pipeline to encode/vectorize the data
	stages = predictRev.encode_vectorize(categoricalColumns, numericCols)
	dataset = predictRev.create_pipeline_train(stages, df, df.columns)
	dataset_trunc = predictRev.remove_outliers(dataset)
	trainingData, testData = predictRev.split_data(dataset_trunc)

	# build linear regression model
	print("\n Linear Regression: \n")
	linreg_model = predictRev.build_linreg(trainingData, testData)

	# extract coefficients by feature from linear regression model
	attrs = sorted((attr["idx"], attr["name"]) for attr in (chain(*dataset
			.schema["features"]
			.metadata["ml_attr"]["attrs"].values())))

	coef_table = pd.DataFrame([[name, linreg_model.coefficients[idx]] for idx, name in attrs])
	coef_table.columns = ['feature', 'coefficient']
	coef_table = coef_table.reindex(coef_table.coefficient.abs().sort_values(ascending=False).index)

	coef_table.to_csv("linreg_coefficients.csv", index=False)

	# build gradient boosted trees model
	print("\n GBT: \n")
	gbt_model = predictRev.build_gbt(trainingData, testData)

	# extract feature importance by feature from gbt model
	feat_imp = pd.DataFrame([[name, gbt_model.featureImportances[idx]] for idx, name in attrs])
	feat_imp.columns = ['feature', 'importance']
	feat_imp = feat_imp.sort_values(by='importance', ascending=False)

	feat_imp.to_csv("gbt_feature_importance.csv", index=False)

	# predict on new data
	df_new = predictRev.read_new_data()
	df_new = predictRev.data_transformation(df_new)

	# encode/vectorize the new data based on training model
	stages_new = predictRev.encode_vectorize(categoricalColumns, numericCols)
	dataset_new = predictRev.create_pipeline_predict(stages, df, df_new, df_new.columns)

	predictRev.fit_linreg(linreg_model, dataset_new).toPandas().to_csv("SalesbyHour_predict_linreg.csv")
	predictRev.fit_gbt(gbt_model, dataset_new).toPandas().to_csv("SalesbyHour_predict_gbt.csv")

if __name__ == "__main__":

	main()