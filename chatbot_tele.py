import re
import nltk
from util import JSONParser
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(chat):
    chat = chat.lower()
    chat = re.sub(r'[^a-zA-Z0-9\s]', '', chat)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    chat = stemmer.stem(chat)

    chat = word_tokenize(chat)
    chat = [kata for kata in chat if kata not in stopwords.words('indonesian')]
    return chat

path = 'data/intents.json'
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

df['chat_preprocess'] = df.chat.apply(preprocess)