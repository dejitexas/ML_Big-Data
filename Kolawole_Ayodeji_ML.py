# Databricks notebook source
# MAGIC %md
# MAGIC #### Importing Required Modules and Libraries

# COMMAND ----------


# Modules for analysis
import numpy as np
import pandas as pd

#PySpark modules
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import udf
from pyspark.sql.functions import isnan, when, count, col

# Spark ML modules
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.pipeline import Pipeline

from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import BinaryClassificationMetrics


#Visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns




# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading the data, cleaning and analyzing the data

# COMMAND ----------

FaultDataset = spark.read.format('csv')\
    .option('inferSchema','true')\
    .option('header','true')\
    .option('sep',',')\
    .load('dbfs:/FileStore/tables/FaultDataset.csv')

# COMMAND ----------

# Checking total number of entries

print('Total entries in the dataframe = ', FaultDataset.count())

# COMMAND ----------

FaultDataset.groupBy('fault_detected').count().display()

# COMMAND ----------

print(FaultDataset.printSchema())


# COMMAND ----------

# MAGIC %md
# MAGIC All the column have data type fload.So, no schema change required.

# COMMAND ----------

# checking the null or nan values:
missing_count = FaultDataset.select([count(when(isnan(n) | col(n).isNull(), n)).alias(n) for n in FaultDataset.columns])
missing_count = missing_count.toPandas().T.reset_index()
missing_count.columns = ['Attributes', 'Number of Null and NaN']
missing_count

# COMMAND ----------

# MAGIC %md
# MAGIC The data set don't contain any missing value

# COMMAND ----------

# Discriptive statistics for each column
desc_stat = FaultDataset.describe().toPandas()
desc_stat.set_index('summary').T

# COMMAND ----------

# Discriptive statistics for fault_detected column
FaultDataset.select('fault_detected').describe().show()

# COMMAND ----------

# Function to plot histogram and box plot
def Plot_col(input_df, col_imput):
  df = input_df
  colum = col_imput
  x = [n[0] for n in df.select(f'{colum}').sample(False, 0.2).collect()]

  #plotting boxplot to see the distribution of number of shares
  fig = plt.figure(figsize = (8,7))
  ax = plt.gca()
  ax.boxplot(x, vert=True, widths = 0.3) 
  ax.set_ylabel(f'{colum}', fontsize = 20)
  ax.set_title(f'Box Plot: Distribution of number of {colum}', fontsize = 20)
  plt.show()


  # plotting histogram to see the frquency distribution of number of shares
  count, bin_edges = np.histogram(x)
  plt.figure(figsize = (12,8))
  ax = sns.distplot(x, hist=True, kde=True, 
               bins=int(180/5), color = '#2038d6',
               hist_kws={'edgecolor':'black'},
               kde_kws={'linewidth': 4})

  ax.set_title(f'Histogram: Distribution of number of {colum}', fontsize = 15)
  ax.set_xlabel(f'No. of {colum}', fontsize = 15)
  ax.set_ylabel('Frequency', fontsize = 15)
  ax.set_xticks(bin_edges)
  plt.show()




# COMMAND ----------

# Calling the funcitons
Plot_col(FaultDataset,'fault_detected')

# COMMAND ----------

# Bar plot: 
fault_count= FaultDataset.select( 'fault_detected').groupBy('fault_detected').count().toPandas()

plt.figure(figsize = [10,8])
sns.barplot(x = fault_count['fault_detected'], y=fault_count['count'], palette=['#9fa1ed', '#2038d6'], edgecolor='black')
plt.title('fault_detected as 1:Y and fault_detected as 0:No ',  fontsize = 16)
plt.show()




# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Engineering

# COMMAND ----------


# Removing fault_detected column and taking rest as features
feature_col = FaultDataset.columns[:-1]
print('Total number of features = ', len(feature_col))

#Converting all features to a dense vector column called features
vectorAssembler = VectorAssembler(inputCols = feature_col, outputCol = 'features')

#Normalizing the new features column
normalizer = Normalizer(inputCol = 'features', outputCol = 'features_normalized')


# COMMAND ----------

#Featuring engineering pipelines
feature_pipe = Pipeline(stages = [vectorAssembler,normalizer ])

# COMMAND ----------

# Transforming the dataset
df_transformed =feature_pipe.fit(FaultDataset).transform(FaultDataset)
print(f'Number of rows in transformed data set {df_transformed.count()}')

# COMMAND ----------

#Check what index was assigned to the labels
df_transformed = df_transformed.selectExpr('fault_detected as label','features','features_normalized')


# COMMAND ----------

# Shuffle Data
df_transformed = df_transformed.orderBy(F.rand())

# Train Test split : Data is split into Training data, Testing data and Data to be used for prediction
df_train, df_test, df_predict = df_transformed.randomSplit([0.7, 0.2997, 0.0003], seed= 101)


#Let's check the size of all the three (without pca)
print(f'Size of training data is {df_train.count()}')
print(f'Size of testing data  is {df_test.count()}')
print(f'Size of data to be used for prediction is {df_predict.count()}')
print('')


# COMMAND ----------

# MAGIC %md
# MAGIC #### Building Classification Models <br>
# MAGIC Algo used:
# MAGIC - RandomForestClassifier 
# MAGIC - Gradient Boosted Tree Classifier

# COMMAND ----------

# Initialization of classifiers 

#Building RandomForestClassifier 
rfc = RandomForestClassifier(featuresCol='features_normalized', labelCol='label', predictionCol='prediction', numTrees = 150)


#Building Gradient Boosted Tree Classifier
gbt = GBTClassifier(featuresCol='features_normalized',labelCol='label', maxIter=40)




# COMMAND ----------

# MAGIC %md
# MAGIC #### Training the classifier

# COMMAND ----------

# Training the Random Forest model with featurea 
rfc_model = rfc.fit(df_train)


# COMMAND ----------

# Training the GBT model with featurea 
gbt_model = gbt.fit(df_train)



# COMMAND ----------

# MAGIC %md
# MAGIC #### Model evaluation and best model confirmation <br>
# MAGIC Evaluating the Machine Learning Models on test data

# COMMAND ----------

#Initializing the evaluation metric: Area Under ROC
evalu_rfc_binary = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC',rawPredictionCol='rawPrediction' )
evalu_gbt_binary = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC',rawPredictionCol='rawPrediction')


#Initializing the evaluation metric: Accuracy 
evalu_rfc_multi = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy',predictionCol='prediction')
evalu_gbt_multi = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy',predictionCol='prediction')


# RandomForest Prediction
rfc_prediction = rfc_model.transform(df_test)

#GBT prediction
gbt_prediction = gbt_model.transform(df_test)


#Calculating AUC and Accuracy for Random Forest model
print(f'AUC for Ramdomforest classifier model is {evalu_rfc_binary.evaluate(rfc_prediction): .3f}')
print(f'Accuracy for Ramdomforest classifier model is {evalu_rfc_multi.evaluate(rfc_prediction): .3f}')


print('')
print('')

#Calculating AUC and Accuracy for Gradient Boosted Tree model
print(f'AUC for GradientBoostedTree classifier model is {evalu_gbt_binary.evaluate(gbt_prediction): .3f}')
print(f'Accuracy for GradientBoostedTree classifier model is {evalu_gbt_multi.evaluate(gbt_prediction): .3f}')





# COMMAND ----------

# MAGIC %md
# MAGIC ##### Creating function to plot the metices

# COMMAND ----------

def plot_metrics(model1_predict, model2_predict):


  rfc_prediction = model1_predict
  gbt_prediction = model2_predict
  
  #Creating class for visualization AUC curve as pyspark don't provide it out-of-the-box for rf and gbt
  class CurveMetrics(BinaryClassificationMetrics):
      def __init__(self, *args):
          super(CurveMetrics, self).__init__(*args)

      def _to_list(self, rdd):
          points = []
          for row in rdd.collect():
              points += [(float(row._1()), float(row._2()))]
          return points

      def get_curve(self, method):
          rdd = getattr(self._java_model, method)().toJavaRDD()
          return self._to_list(rdd)



  #Random forest 
  preds_rfc = rfc_prediction.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
  points_rfc = CurveMetrics(preds_rfc).get_curve('roc')


  #GBT classifier
  preds_gbt = gbt_prediction.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
  points_gbt = CurveMetrics(preds_gbt).get_curve('roc')



  ###Plotting AUC
  plt.figure(figsize=(12,10))

  #RF
  x_rfc = [x[0] for x in points_rfc]
  y_rfc = [x[1] for x in points_rfc]


  #GBT
  x_gbt = [x[0] for x in points_gbt]
  y_gbt = [x[1] for x in points_gbt]



  plt.title('Comparing AUC for RF and GBT', fontsize=15)
  plt.xlabel('False Positive Rate(1-Specificity)', fontsize=15)
  plt.ylabel('True Positive Rate(Sensitivity)', fontsize=15)
  plt.plot(x_rfc, y_rfc, color = 'green', label = f'Random Forest Classifier : AUC = {evalu_rfc_binary.evaluate(rfc_prediction): .3f}')
  plt.plot(x_gbt, y_gbt, color= 'red', label = f'GBT Classifier : AUC = {evalu_gbt_binary.evaluate(gbt_prediction): .3f}')

  plt.legend()
  plt.show()

# COMMAND ----------

# Calling the plot function
plot_metrics(rfc_prediction, gbt_prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tuning the best classifier
# MAGIC So, the best predictor is **GRADIENT BOOSTED TREE CLASSIFIER (GBTClassifier)**. Tuning it further for better prediction

# COMMAND ----------

#Initializing the ParamGridBuilder
grid = ParamGridBuilder().addGrid(gbt.maxDepth, [5,10, 20])\
                         .addGrid(gbt.maxBins, [15,30,60])\
                         .addGrid(gbt.maxIter, [5,10, 20])\
                         .build()
            
#Initializing evaluator
evaluator = MulticlassClassificationEvaluator( metricName='accuracy')

#Initiazising the cross validator
cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=evaluator, parallelism=2, numFolds=5)

#Fitting the model
gbt_model = cv.fit(df_transformed)

#Saving the model
gbt_model.save('best_model.h5')
