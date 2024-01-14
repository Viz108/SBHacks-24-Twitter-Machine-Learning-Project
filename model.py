import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline

# data = pd.read_csv("Twitterdatainsheets.csv", dtype={'index': int, 'Tweet-ID': 'string', 'Weekday': 'string', 'Hour': int, 'Day': int, 'Lang': 'string', 'IsReshare': bool, 'Reach': float, 'RetweetCount': float, 'Likes': float, 'Klout': float, 'Sentiment': float, 'text': 'string', 'LocationID': float, 'UserID': 'string'})

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('Twitterdatainsheets.csv')


df.columns = df.columns.str.strip()

missing_values = df.isnull().sum()

data_types = df.dtypes
data = df[['Weekday', 'Hour', 'IsReshare', 'Reach', 'RetweetCount', 'Likes']]
data = data.dropna(subset=['Weekday', 'Hour', 'IsReshare', 'Reach', 'RetweetCount', 'Likes'])

weekdayEncoder = LabelEncoder()
weekdayEncoder.fit(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
data["Weekday"] = weekdayEncoder.transform(data['Weekday'])

weekdayEncoder = LabelEncoder()
weekdayEncoder.fit(["False", "True"])
data["IsReshare"] = weekdayEncoder.transform(data['IsReshare'])

# # print(data.shape)
text_data = df['text']
text_data = text_data.dropna()
# # print(text_data['text'])

# sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer = "cardiffnlp/twitter-roberta-base-sentiment-latest")
# print(sentiment_pipeline("FUCK YOU!!!"))










from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)
model.save_pretrained(MODEL)
sentiment_list =[]
i=0
for text in text_data[:100000]:
    i+=1
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    print(i, text)
    l = config.id2label[ranking[0]]
    sentiment_list.append(l)

data["Sentiment"] = sentiment_list
print(len(data["Sentiment"]))

weekdayEncoder = LabelEncoder()
weekdayEncoder.fit(["Negative", "Neutral", "Positive"])
data["Sentiment"] = weekdayEncoder.transform(data['Sentiment'])


# for sent in sentiment_list:
#     print(sent)


# specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest")
# print(specific_model(data))