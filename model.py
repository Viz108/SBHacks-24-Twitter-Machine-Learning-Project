import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# data = pd.read_csv("Twitterdatainsheets.csv", dtype={'index': int, 'Tweet-ID': 'string', 'Weekday': 'string', 'Hour': int, 'Day': int, 'Lang': 'string', 'IsReshare': bool, 'Reach': float, 'RetweetCount': float, 'Likes': float, 'Klout': float, 'Sentiment': float, 'text': 'string', 'LocationID': float, 'UserID': 'string'})

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('Twitterdatainsheets.csv')
print(df.info)

print(df.columns)

df.columns = df.columns.str.strip()
print(df.columns)

missing_values = df.isnull().sum()

data_types = df.dtypes

# Display the results
print("Missing Values:")
print(missing_values)

print("\nData Types:")
print(data_types)


print(df['Day'])