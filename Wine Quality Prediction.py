# Importing Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# Data Collection

wine_dataset = pd.read_csv('winequality.csv')
wine_dataset.shape
wine_dataset.head()


# Checking for missing values 

wine_dataset.isnull().sum()


# Data Analysis and Visualization

wine_dataset.describe()


# Number of values for each quality

sns.catplot(x="quality", data = wine_dataset, kind = "count")


# Volatile acidity Vs Quality

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data = wine_dataset)


# Citric acid Vs Quality

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data = wine_dataset)


# Correlation : Positive & Negative Correlation 

correlation = wine_dataset.corr()


# Constructing a heatmap to understand the correlation between the columns

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = ".1f", annot = True, annot_kws={"size":8}, cmap = "Blues")


# Data Preprocessing

X = wine_dataset.drop("quality", axis=1)



print(X)

# Label Binarization or Encoding

y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)

print(y)

# Data Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


print(y.shape, y_train.shape, y_test.shape)
print(X.shape, X_train.shape, X_test.shape)

# Model Training

model = RandomForestClassifier()
model.fit(X_train, y_train)


# Model Evaluation : accuracy score

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print("Accuracy : ", test_data_accuracy)


# Building a Predictive System

input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)


# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)


# reshape the data array as we are predicting th label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 1):
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")

