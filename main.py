from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

path_csv = 'spam.csv'
msgs = pd.read_csv(path_csv, encoding='latin-1')
msgs.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
msgs = msgs.rename(columns={'v1': 'class', 'v2': 'text'})

print(msgs.head())