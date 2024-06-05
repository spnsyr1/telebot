import logging
import json
import random
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Download necessary NLTK data
nltk.download('punkt')

# Define stop words for Indonesian
stop_words_indonesia = set(nltk.corpus.stopwords.words('indonesian'))

# Load dataset
with open("./data/intents.json", "r") as file:
    data = json.load(file)

# Preprocess dataset
patterns = []
responses = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        responses.append(random.choice(intent["responses"]))  # Memilih respon acak

df = pd.DataFrame({"Pertanyaan": patterns, "Jawaban": responses})

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.stop_words = stop_words_indonesia
    
    def preprocess(self, text):
        # Normalisasi (ubah menjadi huruf kecil)
        text = text.lower()
        # Tokenisasi
        tokens = nltk.word_tokenize(text)
        # Hapus stop words dan stemming
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [self.preprocess(text) for text in X]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Pertanyaan'], df['Jawaban'], test_size=0.2, random_state=42)

# Integrasikan ke pipeline
pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Preprocess and fit the data
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

prev_answer = ""


# Method to predict answer based on user input
def predict_answer(user_input):
    return pipeline.predict([user_input])[0]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Halo! Selamat datang di Bot Pengenal UUD 1945. Silahkan ketikkan pasal UUD 1945 berapa yang ingin anda cari :D"
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global prev_answer
    text = update.message.text

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=predict_answer(text)
    )


if __name__ == "__main__":
    application = (
        ApplicationBuilder()
        .token("7176532833:AAHm638Ky41MZSoGmiTYr_d--i_UoLdP74w")
        .build()
    )

    start_handler = CommandHandler("start", start)
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)

    application.add_handler(start_handler)
    application.add_handler(echo_handler)

    application.run_polling()