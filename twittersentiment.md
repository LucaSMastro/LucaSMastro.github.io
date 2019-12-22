**Twitter Sentiment Analysis:** For this project, we will attempt to build a model to determine whether tweets are positive or negative. Given the breadth of subjects addressed in tweets, such a model will be well-suited for general use across different subjects. Although the cost of this generalization is apt to be a lack of power when compared to more specialized models, the ease-of-applicability of such a model is valuable in that it can be utilized to gain a rudimentary big-picture view of user sentiment prior to the development of a specialized model for issues relating to a particular industry or organization.

### 1. Loading the data

For this project, we are using the Sentiment140 dataset from Kaggle. It contains 1.6 million tweets with accompanying scores ranking sentiment from 0-4, negative to positive. It can be retrieved [here.](https://www.kaggle.com/kazanova/sentiment140/data)

```python
import pandas as pd
import numpy as np

tweetsdf = pd.read_csv('C:/datasets/training.1600000.processed.noemoticon.csv', encoding='iso-8859-1')
#tweetsdf = tweetsdf.sample(frac=0.15, random_state=33) #Used to reduce expense when validating model.
print(tweetsdf.head(10))
```

### 2. Formating features

Many of our features contain both references to other users and links to external webpages. These are apt to be almost unique and each have little bearing on sentiment. As a result, we iterate over the dataframe and remove any words starting with either "@" or "http."

```python
#Remove tags and links.
for index, row in tweetsdf.iterrows():
    words = tweetsdf.loc[index, "Text"].split()
    for word in words:
        if word[0]=="@" or word[0:3]=="http":
            words.remove(word)
    text = " ".join(words)
    tweetsdf.loc[index, "Text"] = text

print(tweetsdf.head(10))
```

Afterwards, it remains to format our features into a vector for training a model. We will use a TF-IDF, a frequent choice for sentiment analysis which is able to weight words based on importance.

```python
#Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2500)
features = tweetsdf['Text']
features = vectorizer.fit_transform(features)
```

### 3. Model Selection

For this project, we will use scikit-learn to train a random forest model and a logistic regression model then test to find which one gives higher accuracy.

```python
#Train a Random Forest Classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(features, tweetsdf['Sentiment'], test_size=0.25, random_state=33)
model = RandomForestClassifier(n_estimators=100, random_state=33)

model.fit(x_train, y_train)

prediction=model.predict(x_test)
print(accuracy_score(y_test, prediction))
#Accuracy score of 0.83012
```

Next, we will test the accuracy of a logistic regresison model.

```python
#Comparing with Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=250, random_state=33)

model.fit(x_train, y_train)

prediction=model.predict(x_test)
print(accuracy_score(y_test, prediction))
#Accuracy score of 0.84222
#Logistic Regression performs approximately with about 7.1% higher relative accuracy than a Random Forest.
```

Our logistic regression performs significantly better than our random forest for this task, successfully classifying 84.2% of tweets.

### 5. Discussion

  Such a model could be utilized in a professional setting to gain a quick understanding of how customers are reacting to an organizational initiative. A company could select tweets featuring their name from before and after some initiative to see how average user sentiment changes. Alernatively, this model could also be used to gauge outreach to diverse groups. By filtering tweets referencing the organization by the demographic information of the posters, it would be possible to quickly determine which groups are being attracted by initiatives and which are being alienated.
