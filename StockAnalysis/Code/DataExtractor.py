
"""
Created on Tue Nov 16 16:29:56 2021

@author: om160
"""

import twint
import nest_asyncio
import pandas as pd
nest_asyncio.apply()
import flair
sentiment_model = flair.models.TextClassifier.load('en-sentiment')
import pandas as pd
import re
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())
def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
         if w.lower() in words or not w.isalpha())
    return tweet
import datetime
from datetime import timedelta
from statistics import mean
import numpy as np
input = '2021/11/10'
format = '%Y-%m-%d %H:%M:%S'

today = datetime.datetime.now()
yday = today- timedelta(days = 1)
yday = yday.strftime("%Y-%m-%d %H:%M:%S")
today = today.strftime("%Y-%m-%d %H:%M:%S")
size = 1000
counter = 2
dates = []
scores = []
while size:
  print(today)
  print(yday)
  c = twint.Config()
  c.Lang = 'en'
  c.Search = "$tesla"
  c.Popular_tweets = True
  c.Hide_output = True
  c.Pandas = True
  c.Since = yday
  c.Until = today
  c.Storage = True
  c.Lang = 'en'
  c.Limit = 100
  twint.run.Search(c)
  Tweets_df = twint.storage.panda.Tweets_df
  try:
    df = Tweets_df.loc[Tweets_df['language'] == 'en']
  except:
    scores.append(np.nan)
    
    today = yday
    yday = datetime.datetime.now() - timedelta(days = counter)
    counter = counter + 1
    yday = yday.strftime("%Y-%m-%d %H:%M:%S")
    size= size-1
    dates.append(datetime.datetime.strptime(today, '%Y-%m-%d %H:%M:%S').date())
    continue
  size = size-1
  today = yday
  yday = datetime.datetime.now() - timedelta(days = counter)
  counter = counter + 1
  yday = yday.strftime("%Y-%m-%d %H:%M:%S")
  probs = []
  
  print(counter)
  df['tweet'] = df['tweet'].map(lambda x: cleaner(x))
  for tweet in df['tweet'].to_list():
    # make prediction
    sentence = flair.data.Sentence(tweet)
    sentiment_model.predict(sentence)
    try:
      # extract sentiment prediction
      
      if(sentence.labels[0].value == "NEGATIVE"):
        probs.append(-1*sentence.labels[0].score)  
      else:
        probs.append(sentence.labels[0].score)  
    except IndexError:
      probs.append(0)
  scores.append(mean(probs))
  dates.append(datetime.datetime.strptime(today, '%Y-%m-%d %H:%M:%S').date())
final_df = pd.DataFrame(list(zip(dates, scores)),
               columns =['Date', 'Score'])
final_df.isna().sum()

final_df = final_df.interpolate()
final_df
import yfinance as yf
ticker = yf.Ticker('TSLA')
nsf = ticker.history(start="2019-02-23", end="2021-11-15", interval="1d")
nsf = nsf.drop("Dividends",axis =1)
nsf = nsf.drop("Volume", axis = 1)
nsf = nsf.drop("Stock Splits", axis = 1)
nsf

nsf
final_df = final_df.set_index('Date')
final_df = final_df.merge(nsf, left_index=True, right_index=True, how='inner')
cat = []
for i in range(len(final_df['Close'])-1):
  if(final_df['Close'][i]<final_df['Close'][i+1]):
    cat.append(1)
  else:
    cat.append(-1)

cat.append(0)
final_df['label']=cat

final_df.to_csv("final_dataset_tsla.csv")
