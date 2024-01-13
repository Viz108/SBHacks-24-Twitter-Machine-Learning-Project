import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures

#data = pd.read_csv("Twitterdatainsheets.csv", dtype={'index': int, 'TweetID': 'string', ' Weekday': 'string', ' Hour': int, ' Day': int, ' Lang': 'string', ' IsReshare': bool, ' Reach': float, ' RetweetCount': float, ' Likes': float, ' Klout': float, ' Sentiment': float, ' text': 'string', ' LocationID': float, ' UserID': 'string'})
data = pd.read_csv("Twitterdatainsheets.csv", low_memory=False)

print(data)
data.columns = data.columns.str.strip() 
print(data.columns.tolist())

data = data[['Weekday', 'Hour', 'IsReshare', 'Reach', 'RetweetCount', 'Likes']]
print(data.shape)
data = data.dropna(subset=['Weekday', 'Hour', 'IsReshare', 'Reach', 'RetweetCount', 'Likes'])
print(data.shape)

predict = [' Reach', ' RetweetCount', ' Likes']

weekdayEncoder = LabelEncoder()
weekdayEncoder.fit(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
data['Weekday'] = weekdayEncoder.transform(data['Weekday'])

reshareEncoder = LabelEncoder()
reshareEncoder.fit(["False", "True"])
data['IsReshare'] = reshareEncoder.transform(data['IsReshare'])

print(data.head())

labels = data[['Weekday', 'Hour', 'IsReshare']]
features = data[['Reach', 'RetweetCount', 'Likes']]


x_train, x_test, y_train, y_test = train_test_split(labels, features, test_size = 0.1)

model = linear_model.PoissonRegressor()
multioutputmodel = MultiOutputRegressor(model).fit(x_train, y_train)
print(multioutputmodel.score(x_test, y_test))
print(multioutputmodel.predict(x_test))

