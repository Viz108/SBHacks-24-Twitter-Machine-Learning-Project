import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#data = pd.read_csv("Twitterdatainsheets.csv", dtype={'index': int, 'TweetID': 'string', 'Weekday': 'string', 'Hour': int, 'Day': int, 'lang': 'string', 'IsReshare': bool, 'Reach': float, 'RetweetCount': float, 'Likes': float, 'Klout': float, 'Sentiment': float, 'text': 'string', 'LocationID': float, 'UserID': bool})
data = pd.read_csv("Twitterdatainsheets.csv", low_memory=False)

print(data)
print(data.columns.tolist())
print(data[' Day'])

