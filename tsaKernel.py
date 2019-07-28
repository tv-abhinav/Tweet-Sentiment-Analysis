import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
gnb = GaussianNB()
trainCSV = pd.read_csv("train.csv", encoding="ISO-8859-1")
df = pd.DataFrame(trainCSV)
df = df.drop(['ItemID'], axis=1)
# column_names=["SentimentText","Sentiment"]
# df=df.reindex(columns=column_names)
# df.rename(columns={'SentimentText':'Tweet','Sentiment':'Result'},inplace=True)
le.fit(df['SentimentText'])
n_tweet = le.transform(df['SentimentText'])
Result = df['Sentiment']
Ypred = gnb.fit(n_tweet,Result).predict(Result)