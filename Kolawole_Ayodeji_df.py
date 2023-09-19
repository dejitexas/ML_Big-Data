# Databricks notebook source
# MAGIC %md
# MAGIC ## Task 1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Solution: PySpark

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Importing required libraries

# COMMAND ----------

from pyspark.sql.functions import col,year,substring,trim,lit, count_distinct, desc

# COMMAND ----------

# MAGIC %md
# MAGIC Reading <code>clinicaltrial_2021.csv</code>

# COMMAND ----------



df_clinicaltrial = spark.read.format('csv')\
    .option('inferSchema','true')\
    .option('header','true')\
    .option('sep','|')\
    .load('dbfs:/FileStore/tables/clinicaltrial_2021.csv').withColumn('file_year',lit('2021'))

# COMMAND ----------

# MAGIC %md
# MAGIC Reading <code>pharma.csv</code>

# COMMAND ----------

df_pharma = spark.read.format('csv')\
    .option('inferSchema','true')\
    .option('header','true')\
    .option('sep',',')\
    .load('dbfs:/FileStore/tables/pharma.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 1. The number of studies in the dataset. You must ensure that you explicitly check distinct studies.

# COMMAND ----------


df_clinicaltrial.select(col('id'),col('file_year')).groupBy('file_year').count().orderBy('file_year').display()


# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. You should list all the types (as contained in the Type column) of studies in the dataset along with the frequencies of each type. These should be ordered from most frequent to least frequent.

# COMMAND ----------

df_clinicaltrial.select(col('Type'),col('file_year')).groupBy(col('Type'))\
                .pivot('file_year').count().orderBy(col('2021').desc()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3. The top 5 conditions (from Conditions) with their frequencies.

# COMMAND ----------

from pyspark.sql.functions import split,explode
df_clinicaltrial.select(col('file_year'),explode(split(col('Conditions'),',')).alias('Conditions')).groupBy('Conditions').pivot('file_year').count().orderBy(col('2021').desc()).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4. Find the 10 most common sponsors that are not pharmaceutical companies, along with the number of clinical trials they have sponsored. Hint: For a basic implementation, you can assume that the Parent Company column contains all possible pharmaceutical companies.

# COMMAND ----------

df_clinicaltrial.join(df_pharma, df_clinicaltrial['Sponsor']==df_pharma['Parent_Company'],'anti').select('Sponsor','file_year').groupBy('Sponsor').pivot('file_year').count().select(col('Sponsor'),col('2021')).orderBy(col('2021').desc()).show(10,False)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5. Plot number of completed studies each month in a given year – for the submission dataset, the year is 2021. You need to include your visualization as well as a table of all the values you have plotted for each month.

# COMMAND ----------

df_month = spark.createDataFrame(data=[('Jan',1),('Feb',2),('Mar',3),('Apr',4),('May',5),('Jun',6),('Jul',7),('Aug',8),('Sep',9),('Oct',10),('Nov',11),('Dec',12)],schema=['Month','Month_no']).withColumnRenamed('Month','Month_')


# COMMAND ----------

from pyspark.sql.functions import to_date, lit, concat,year,date_format


  
df_month_2021 = df_clinicaltrial.select('*').withColumn('Completion',to_date(concat(lit('01'),col('Completion')),'ddMMM yyyy'))\
                                .where(col('Status')=='Completed')\
                                .where( (year(col('Completion')) == '2021')  ).filter("file_year in ('2021')")\
                                .withColumn('Month',date_format(col('Completion'),'MMM'))

df_month_2021 = df_month_2021.select('Month','file_year').groupBy('Month').pivot('file_year').count()



df_month_2021 = df_month_2021.join(df_month, df_month_2021['Month']==df_month['Month_'],'inner')\
                             .orderBy('Month_no').select('Month','2021')
df_month_2021.display()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

months = [df_month_2021.select(col('Month')).collect()[i][0] for i in range(0,10)]
values = [df_month_2021.select(col('2021')).collect()[i][0] for i in range(0,10)]
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(months, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Months")
plt.ylabel("No. of completed studies")
plt.title("Completed studies each month in a given year 2021")
plt.show()
