import pandas as pd
import matplotlib.pyplot as plt
from sklearn import 
import numpy as np

trainCSV = pd.read_csv("train.csv", encoding="ISO-8859-1")
df = pd.DataFrame(trainCSV)
df = df.drop(['ItemID'], axis=1)
column_names=["SentimentText","Sentiment"]
df=df.reindex(columns=column_names)
df.rename(columns={'SentimentText':'Tweet','Sentiment':'Result'},inplace=True)
print(np.count_nonzero(df.duplicated(subset="Tweet", keep='first')))
print(df.head(5))