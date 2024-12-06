# This tool uses VADER model for analyzing Sentiments of reviews given by customers
# This program was already written in the Databricks environment

# Installing all the libraries
!pip install pandas
!pip install langdetect
!pip install nltk
!pip install numpy
!pip install tqdm
!pip install transformers
!pip install torch

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, MapType, TimestampType
# from pyspark.sql.types import *
# from pyspark.sql.functions import *
import zipfile
import pandas as pd
import numpy as np
from pyspark.sql.functions import col
import os
from pyspark.sql.functions import lit, to_date, to_timestamp
from pyspark.sql.functions import input_file_name

path = "/mnt/adls/container/PrachiSingh/SentAnalysis/Hotel/Hotel_Review_Rating.xlsx"

df = spark.read\
          .format("com.crealytics.spark.excel")\
          .option("header", "true")\
          .option("treatEmptyValuesAsNulls", "true")\
          .load(path)
df.printSchema()

# Converting PySpark Dataframe to Pandas Dataframe
df = df.toPandas()
# Handling Null Values
df_sum = df.isnull().sum().to_frame()
df_sum.display()
# Using the df_sum dataframe, remove the columns where nulls are more than 30% of the records
df = df.drop(df.columns[12,13]],axis=1)
# Filling the nulls for the rest of the columns, if any
df = df.interpolate(method="linear",limit_direction="forward)
# Make sure that first row of the dataframe does not contain any nulls, in order to use interpolate method.

# Changing Yes/No to 1/0 for easier EDA
df = df.replace(to_replace="Yes",value=1)
df = df.replace(to_replace="No",value=0)
df.info()

# Preparing dataframe of list of all columns
df_colnames = df.columns.to_frame()
df_colnames.display()

df[["RateOnSacaleOf1_10"]] = df[["RateOnSacaleOf1_10"]].apply(pd.to_numeric)
df = df.sort_values(by=['RateOnSacaleOf1_10'],ascending=True)

# Importing more modules
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import nltk.tag import pos_tag
import seaborn as sns

nltk.download('punkt')
nltl.download('stopwords')

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

df[["Review"]] = df[["Review"]].astype(str)

# Running the polarity score on the entire dataset
for i,row in tqdm(df.iterrows(),total=len(df.columns)):
  text = row['Review']
  myid = row['ID_Customer']
  res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(colums={'index':'ID_Customer'})
vaders = vaders.merge(df,how='left)
vaders.display()

# Getting graphs for positive, negative, neutral, and compound score obtained through VADER model against the Rating column
sns.barplot(x='RateOnSacaleOf1_10',
            y='pos',
            data=vaders)
plt.xlabel("Rating")
plt.ylabel("Positive Score from VADER")
plt.show()
sns.barplot(x='RateOnSacaleOf1_10',
            y='neu',
            data=vaders)
plt.xlabel("Rating")
plt.ylabel("Neytral Score from VADER")
plt.show()
sns.barplot(x='RateOnSacaleOf1_10',
            y='neg',
            data=vaders)
plt.xlabel("Rating")
plt.ylabel("Negative Score from VADER")
plt.show()
sns.barplot(x='RateOnSacaleOf1_10',
            y='compound',
            data=vaders)
plt.xlabel("Rating")
plt.ylabel("Compound Score from VADER")
plt.show()
# s
# Function for Sentiment Type
def f(df):
  if df['compound'] >= 0.5:
    val = 1 #Positive Sentiment
  elif df['compound'] > -0.5 & df['compound'] < 0.5:
    val = 2 #Neutral Sentiment
  elif df['compound'] <=-0.5:
    val = 3 #Negative Sentiment
vaders['SentimentType'] = vaders.apply(f,axis=1)

# Advertisement that brought customers to this hotel
# The Ad columns were in Yes/No form
def get_count_df(df):
  list_name=[]
  count_yes=[]
  for c in df.columns:
    list_name.append(c)
    count_yes.append(df[c].value_counts()[1])
  var_key = {'var_name':list_name,'count_value':count_yes}
  new_df = pd.Dataframe(var_key)
  new_df = new_df.sort_values(by=['count_value'],ascending=True)
  return new_df
df_ad = vaders.loc[:,['Billboards_signs','TV_Ads','Print_Ad_Newspaper','Google_Ads','Instagram','Facebook']]
ad_df = get_count_df(df_ad)
list_ad_name = ['Billboards/signs','TV Ads','Print Ad (Newspaper)','Google Ads','Instagram','Facebook']]
ad_df['var_name'] = list_ad_name
ad_df.plot.barh(x='var_name',y='count_value',title='Ads Vs Count of customers')
plt.ylabel('Ads Media')
plt.xlabel('Count of customers')
plt.show()

# Analyzing Ratings given on the basis of age groups, gender, and marital status
df_age_marital_gender = vaders.loc[:,['Age','Gender','MaritalStatus','RateOnSacaleOf1_10']]
# Dividing Age of customers into age groups
def f_agegroup(df):
  if df['Age']>=18 & df['Age']<=25:
    return 'Between 18 and 25'
  elif df['Age']>25 & df['Age']<=35:
    return 'Between 25 and 35'
  elif df['Age']>35 & df['Age']<=50:
    return 'Between 35 and 50'
  elif df['Age']>50:
    return 'Above 50'

df_age_marital_gender['AgeGroup'] = df_age_marital_gender.apply(f_agegroup,axis=1)
df_age_marital_gender = df_age_marital_gender.drop(columns=['Age'])

df_gender = df_age_marital_gender.groupbt(["Gender","RateOnSacaleOf1_10")["RateOnSacaleOf1_10"].value_counts().unstack(fill_value=0)
df_gender.plot(kind="bar", stacked=True)
plt.xticks(rotation=90,horizontalalignment="center")
plt.title("Analysis of the Gender of Customers Vs their Rating scores")
plt.show()

df_age = df_age_marital_gender.groupbt(["AgeGroup","RateOnSacaleOf1_10")["RateOnSacaleOf1_10"].value_counts().unstack(fill_value=0)
df_age.plot(kind="bar", stacked=True)
plt.xticks(rotation=90,horizontalalignment="center")
plt.title("Analysis of the Age groups of Customers Vs their Rating scores")
plt.show()

df_maritalstatus = df_age_marital_gender.groupbt(["MaritalStatus","RateOnSacaleOf1_10")["RateOnSacaleOf1_10"].value_counts().unstack(fill_value=0)
df_maritalstatus.plot(kind="bar", stacked=True)
plt.xticks(rotation=90,horizontalalignment="center")
plt.title("Analysis of the Marital Status of Customers Vs their Rating scores")
plt.show()

# Analysing Room Service
# The RoomService Column contains Poor, Fair, Good, Very Good, and Excellent responses
def get_count_df_good(df):
  list_name=[]
  count_p=[]
  count_f=[]
  count_g=[]
  count_vg=[]
  count_e=[]
  for c in df.columns:
    list_name.append(c)
    if 'Poor' in df[c].unique().tolist():
      count_p.append(df[c].value_counts()['Poor')
    else:
      count_p.append(0)
    if 'Fair' in df[c].unique().tolist():
      count_f.append(df[c].value_counts()['Fair')
    else:
      count_f.append(0)
    if 'Good' in df[c].unique().tolist():
      count_g.append(df[c].value_counts()['Good')
    else:
      count_g.append(0)
    if 'Very Good' in df[c].unique().tolist():
      count_vg.append(df[c].value_counts()['Very Good')
    else:
      count_vg.append(0)
    if 'Excellent' in df[c].unique().tolist():
      count_e.append(df[c].value_counts()['Excellent')
    else:
      count_e.append(0)
    var_key = {'var_name':list_name,'Poor':count_p,'Fair':count_f,'Good':count_g,'Very Good':count_vg,'Excellent':count_e}
    new_df = pd.DataFrame(var_key)
    return new_df
  df_roomservice = vaders.loc[:,['How well do you liked our room service during day','How well do you liked our room service during night']]
  roomservice_df = get_count_df_good(df_roomservice)
  roomservice_df.plot.bar(figsize=(10,12))
  plt.title("Room Service Experience Vs Number of customers")
  plt.xlabel("Experience of Room Service")
  plt.ylabel("Number of customers")
  plt.show()

# Getting the correlation matrix for the whole dataframe
df_corr = vaders.corr(
  method = 'pearson',
  min_periods = 1
)

sns.heatmap(df_corr,annot=True)
plt.show()
