# Tweet-Sentiment-Analysis
This rep contains my experimentation on tweet data to find whether it has a positive, negative or neutral tone. Data set can be found in kaggle whose link is be given below.

## Methodology
Machine Learning Workflow: Problem Statement -> Data Gathering -> Data Formatting -> Algorithm Selection -> Creating Model -> Training Model -> Testing Model -> Repeat till optimum solution

## Problem Statement
Using Tweet Sentiment data after cleaning using Pandas dataframe, to classify the tweets into negative and positive using Gaussian Naive Bayes Algorithm and testing the model using seperate test data and improving accuracy by changing hyper parameters, using cross validation and changing Algorithm if needed.

## Data Set
https://www.kaggle.com/c/twitter-sentiment-analysis2/data -1
Testing vs Training data Ratio: 100K : 300k
### Data fields
ItemID - id of twit
Sentiment - sentiment
SentimentText - text of the twit

0 - negative
1 - positive

-Source 1

## Possible Classification Algorithms
Gaussian Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
Decision Tree Classifier: https://scikit-learn.org/stable/modules/tree.html#classification
Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
(CV version of these)

## Training Data Accuracy
-To be pushed-

## Testing Accuracy
-To be pushed-
