##############################
#       NPL Project          #
##############################
### Load libraries and modules ###
# Dataframes and matrices ----------------------------------------------
import pandas as pd
import numpy as np
import regex as re
import nltk
import os
import joblib
# Machine learning -----------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, svm
# Preprocessing --------------------------------------------------------
from sklearn.pipeline import Pipeline
# Metrics --------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score

nltk.download('stopwords')

######################
# Data Preprocessing #
######################
# Loading the dataset
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')
# Create a copy of the original dataset
df = df_raw.copy()
# We remove the duplicated data
df = df[df.duplicated() == False]
def url_cleaner(url):
  # Remove the start of the url, the protocol and the www
  url_clean = re.sub(r'(https://www|https://|http://www|http://)', '', url)
  # Remove the commas from the url
  url_clean = re.sub(',', ' ', url_clean)
  # url in lower case
  url_clean = url_clean.lower()
  # remove special characters
  url_clean = re.sub('(\\W)+',' ',url_clean)
  # remove duplicated words
  url_clean = re.sub(r'\b(\w+)( \1\b)+', r'\1',url_clean)
  # remove multiple space
  url_clean = re.sub(' +', ' ',url_clean)
  # remove punctuation
  url_clean = re.sub('[^a-zA-Z]', ' ', url_clean)
  # remove tags
  url_clean=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",url_clean)
  return url_clean

df['url_clean'] = df['url'].apply(url_cleaner)
# Building a stop words list
stopwords = nltk.corpus.stopwords.words('english')
# Building a function to remove stop words
def remove_stopwords(text):
    text=' '.join([word for word in text.split() if word not in stopwords])
    return text
# removing stop words from url information
df['url_clean'] = df['url_clean'].apply(remove_stopwords)
# Encoding of the target variable
df['is_spam'] = df['is_spam'].map({True : 1, False: 0})

#####################
# Model and results #
#####################
# Create sparse matrix
message_vectorizer = CountVectorizer().fit_transform(df['url_clean'])
# Spliting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(message_vectorizer, df['is_spam'], test_size = 0.45, random_state = 42, shuffle = True)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# The classifier algorithm is using Support Vector Machine.
classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
classifier.fit(X_train, y_train)
# Classifier prediction
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
print("Confusion Matrix\n",cm)
print("\n")
print(classification_report(y_test,y_pred))
print("\n")
print("Support Vector Machine Mean absolute error:", mean_absolute_error(y_test, y_pred))
print("\n")
print('Support Vector Machine Train Accuracy = ',accuracy_score(y_train,classifier.predict(X_train)))
print('Support Vector Machine Test Accuracy = ',accuracy_score(y_test,classifier.predict(X_test)))
print("\n")
print("Support Vector Machine Precision score:",precision_score(y_test, y_pred))
print("\n")
print("Support Vector Machine Recall score:",recall_score(y_test, y_pred))

# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../models/svm.pkl')

joblib.dump(classifier, filename)