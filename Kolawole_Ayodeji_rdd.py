# Databricks notebook source
# MAGIC %md
# MAGIC ## Task 1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Solution: PySpark

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Unzipping clinicaltrial_2021.zip

# COMMAND ----------

dbutils.fs.rm('file:/tmp/clinicaltrial_2021.csv',True)
dbutils.fs.rm('file:/tmp/clinicaltrial_2021.zip',True)

# COMMAND ----------

dbutils.fs.cp('dbfs:/FileStore/tables/clinicaltrial_2021.zip','file:/tmp/')

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip -d /tmp/ /tmp/clinicaltrial_2021.zip

# COMMAND ----------

dbutils.fs.mv('file:/tmp/clinicaltrial_2021.csv','dbfs:/FileStore/tables/')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Unzipping pharma.zip

# COMMAND ----------

dbutils.fs.cp('dbfs:/FileStore/tables/pharma.zip','file:/tmp/')

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip -d /tmp/ /tmp/pharma.zip

# COMMAND ----------

dbutils.fs.mv('file:/tmp/pharma.csv','dbfs:/FileStore/tables/')

# COMMAND ----------

# MAGIC %md
# MAGIC Reading <code>Clinicaltrial_2021.csv</code>

# COMMAND ----------

rddClinicaltrial_2021 = spark.sparkContext.textFile("dbfs:/FileStore/tables/clinicaltrial_2021.csv").map(lambda element: element.split("|"))


header = rddClinicaltrial_2021.first()
rddClinicaltrial_2021 = rddClinicaltrial_2021.filter(lambda row: row != header)


# COMMAND ----------

# MAGIC %md
# MAGIC Reading <code>pharma.csv</code>

# COMMAND ----------

rddPharma = spark.sparkContext.textFile("dbfs:/FileStore/tables/pharma.csv").map(lambda element: element.split(','))


header_p = rddPharma.first()
rddPharma = rddPharma.filter(lambda row: row != header_p)




# COMMAND ----------

# MAGIC %md
# MAGIC ###### 1. The number of studies in the dataset. You must ensure that you explicitly check distinct studies.

# COMMAND ----------

len(set([n[0] for n in rddClinicaltrial_2021.collect()]))


# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. You should list all the types (as contained in the Type column) of studies in the dataset along with the frequencies of each type. These should be ordered from most frequent to least frequent.

# COMMAND ----------

rdd = rddClinicaltrial_2021.map(lambda x: [x[i] for i in [5]])
rdd = rdd.map(lambda x: x[0])
rdd_mapped = rdd.map(lambda x: (x,1))
rdd_mapped.groupByKey().mapValues(sum).map(lambda x:(x[1],x[0])).sortByKey(False).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3. The top 5 conditions (from Conditions) with their frequencies.

# COMMAND ----------

# Using flatMap
rdd = rddClinicaltrial_2021.map(lambda x: [x[i] for i in [7]]).filter(lambda x: x != '')
rdd = rdd.flatMap(lambda x: x[0].split(','))
rdd_mapped = rdd.map(lambda x: (x,1))
rdd_mapped.groupByKey().mapValues(sum).map(lambda x:(x[1],x[0])).sortByKey(False).take(6)[1:]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4. Find the 10 most common sponsors that are not pharmaceutical companies, along with the number of clinical trials they have sponsored. Hint: For a basic implementation, you can assume that the Parent Company column contains all possible pharmaceutical companies.

# COMMAND ----------

rdd_result = rddClinicaltrial_2021.map(lambda x: x[1]).subtract(rddPharma.map(lambda x: x[1]).map(lambda x: x.replace("\"", "")))
rdd_mapped = rdd_result.map(lambda x: (x,1))
rdd_mapped.collect()
rdd_mapped.groupByKey().mapValues(sum).map(lambda x:(x[1],x[0])).sortByKey(False).take(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5. Plot number of completed studies each month in a given year – for the submission dataset, the year is 2021. You need to include your visualization as well as a table of all the values you have plotted for each month.

# COMMAND ----------

rdd_result = rddClinicaltrial_2021.map(lambda x: [x[i] for i in [2,4]]).filter(lambda x: x[0]=='Completed' and x[1].endswith('2021'))
rdd_result = rdd_result.map(lambda x: x[1])

rdd_mapped = rdd_result.map(lambda x: (x,1))
rdd_final = rdd_mapped.groupByKey().mapValues(sum).map(lambda x:(x[1],x[0])).sortByKey(False)
rdd_final.collect()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

months = [n[1][:3] for n in rdd_final.collect()]
values = [n[0] for n in rdd_final.collect()]
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(months, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Months-2021")
plt.ylabel("No. of completed studies")
plt.title("Completed studies each month in a given year 2021 ")
plt.show()
