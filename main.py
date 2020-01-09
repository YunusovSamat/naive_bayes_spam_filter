import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import pandas as pd


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split()
                   if word.lower() not in stopwords.words('english')]
    return clean_words


path_csv = 'spam.csv'
msgs = pd.read_csv(path_csv, encoding='latin-1')
msgs.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
msgs = msgs.rename(columns={'v1': 'class', 'v2': 'text'})

msg_train, msg_test, class_train, class_test = train_test_split(
    msgs['text'], msgs['class'], test_size=0.2)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, class_train)
predictions = pipeline.predict(msg_test)
print(classification_report(class_test, predictions))
