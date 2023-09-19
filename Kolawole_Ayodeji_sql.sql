-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Task 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Solution: SQL

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC
-- MAGIC
-- MAGIC DROP TABLE IF EXISTS clinicaltrial_2021;
-- MAGIC CREATE TABLE IF NOT EXISTS clinicaltrial_2021 
-- MAGIC USING csv 
-- MAGIC OPTIONS ('sep'= '|','Header'='true')
-- MAGIC LOCATION 'dbfs:/FileStore/tables/clinicaltrial_2021.csv';
-- MAGIC
-- MAGIC create or replace temp view clinicaltrial as
-- MAGIC select *, '2021' as file_year from clinicaltrial_2021;
-- MAGIC
-- MAGIC
-- MAGIC DROP TABLE IF EXISTS pharma;
-- MAGIC CREATE TABLE IF NOT EXISTS pharma 
-- MAGIC USING csv 
-- MAGIC OPTIONS ('sep'= ',','Header'='true')
-- MAGIC LOCATION 'dbfs:/FileStore/tables/pharma.csv';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ###### 1. The number of studies in the dataset. You must ensure that you explicitly check distinct studies.

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select file_year, count(id) as count from clinicaltrial
-- MAGIC group by file_year

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##### 2. You should list all the types (as contained in the Type column) of studies in the dataset along with the frequencies of each type. These should be ordered from most frequent to least frequent.

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC with cte as
-- MAGIC
-- MAGIC (
-- MAGIC select * from (
-- MAGIC select type, type as type_, file_year from clinicaltrial) 
-- MAGIC pivot ( count(type_) for  file_year in ('2021'))
-- MAGIC )
-- MAGIC
-- MAGIC select * from cte 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##### 3. The top 5 conditions (from Conditions) with their frequencies.

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC with cte as
-- MAGIC
-- MAGIC (select *, explode(split(Conditions,',')) as Cond from clinicaltrial )
-- MAGIC
-- MAGIC select cond as condition , count(cond) as cnt
-- MAGIC from cte
-- MAGIC group by cond
-- MAGIC order by cnt desc limit 5

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##### 4. Find the 10 most common sponsors that are not pharmaceutical companies, along with the number of clinical trials they have sponsored. Hint: For a basic implementation, you can assume that the Parent Company column contains all possible pharmaceutical companies.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##### For 2021

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC with cte as
-- MAGIC (select Sponsor,file_year
-- MAGIC from clinicaltrial c anti join pharma p
-- MAGIC on c.Sponsor = p.Parent_Company where file_year = 2021),
-- MAGIC cte2 as 
-- MAGIC (select 
-- MAGIC Sponsor,
-- MAGIC sum(case when file_year = 2021 then 1 else 0 end) as year_2021 from cte
-- MAGIC group by Sponsor
-- MAGIC )
-- MAGIC select * from cte2 order by year_2021 desc limit 10

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##### 5. Plot number of completed studies each month in a given year – for the submission dataset, the year is 2021. You need to include your visualization as well as a table of all the values you have plotted for each month.

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC drop table if exists tabMonth;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.rm('dbfs:/user/hive/warehouse/tabmonth',True)

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC create  table tabMonth(
-- MAGIC `month` string,
-- MAGIC `month_no` int) ;
-- MAGIC insert into tabMonth
-- MAGIC values
-- MAGIC ('Jan',1),
-- MAGIC ('Feb',2),
-- MAGIC ('Mar',3),
-- MAGIC ('Apr',4),
-- MAGIC ('May',5),
-- MAGIC ('Jun',6),
-- MAGIC ('Jul',7),
-- MAGIC ('Aug',8),
-- MAGIC ('Sep',9),
-- MAGIC ('Oct',10),
-- MAGIC ('Nov',11),
-- MAGIC ('Dec',12)

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC create or replace temp view plot 
-- MAGIC as
-- MAGIC (with cte as
-- MAGIC (
-- MAGIC select *, to_date(concat('01',Completion),'ddMMM yyyy') as Completion_ from clinicaltrial
-- MAGIC where status = 'Completed'
-- MAGIC ),
-- MAGIC cte2
-- MAGIC (select *, date_format(Completion_,'MMM') as Month from cte where year(completion_) = '2021' and file_year in ('2021'))
-- MAGIC
-- MAGIC select Month, Year_2021
-- MAGIC from tabMonth join (select Month as Month_, count(Sponsor) as Year_2021
-- MAGIC  from cte2
-- MAGIC  group by Month) as t
-- MAGIC  on tabMonth.Month = t.Month_ order by month_no);
-- MAGIC  select * from plot

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_month_2021 = spark.sql('select * from plot')

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col
-- MAGIC import numpy as np
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC months = [df_month_2021.select(col('Month')).collect()[i][0] for i in range(0,10)]
-- MAGIC values = [df_month_2021.select(col('Year_2021')).collect()[i][0] for i in range(0,10)]
-- MAGIC   
-- MAGIC fig = plt.figure(figsize = (10, 5))
-- MAGIC  
-- MAGIC # creating the bar plot
-- MAGIC plt.bar(months, values, color ='maroon',
-- MAGIC         width = 0.4)
-- MAGIC  
-- MAGIC plt.xlabel("Months")
-- MAGIC plt.ylabel("No. of completed studies")
-- MAGIC plt.title("Completed studies each month in a given year 2021")
-- MAGIC plt.show()
