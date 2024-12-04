# Dataset link- https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

# importing libraries
import pandas as pd
import seaborn as sns

# Loading and exploring the dataset
df_dataset = spark.read.option("inferschema","true").option("header","true").csv("location_of_dataset")

display(df_dataset)

df_pandas_dataset = df_dataset.toPandas()

df_pandas_dataset.shape

# count number of nulls
df_pandas_dataset.isna().sum()

# drop column with null values
# axis = 0 for rows and axis = 1 for columns
df_pandas_dataset.dropna(axis=1,inplace=True)

df_pandas_dataset['diagnosis'].value_counts()

# Label Encoding
df_pandas_dataset.dtypes

# Encoding the categorical column, here it is diagnosis column
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df_pandas_dataset.iloc[:,1] = label_encoder.fit_transform(df_pandas_dataset.iloc[:,1].values)

# split dataset and feature scaling
# Splittinf the dataset into dependent and independent datasets
X= df_pandas_dataset.iloc[:,2:].values #independent
Y= df_pandas_dataset.iloc[:,1].values #dependent

# Splitting datasets into training(75%) and test(25%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25)
# The X_test and Y_test are used for testing the models if it is predicting the right output/labels.

# Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #initialising the instance
X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

# Build a Logistic Regression model
# Logistic Regression model is good for classification problems, especially the binary classifications
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train,Y_train)

# make use of training model to make predictions on test data
predictions = classifier.predict(X_test)

# Performance evaluation of the model

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,predictions)
print(cm)
sns.heatmap(cm,annot=True)

# getting accuracy for model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predictions))

print(Y_test)

print(predictions)
