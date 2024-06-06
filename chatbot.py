import logging
import json
import random
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

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

tfidf_vectorizer = TfidfVectorizer()
x = tfidf_vectorizer.fit_transform(df["Pertanyaan"])
y = df["Jawaban"]

# Train a classifier
classifier = MultinomialNB()
classifier.fit(x, y)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Method to predict answer based on user input
def predict_answer(user_input):
    global prev_answer
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    predicted_answer = classifier.predict(user_input_tfidf)[0]
    return predicted_answer


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Halo! Perkenalkan Aku adalah SiPasal, Bot Pengenal UUD 1945! Silahkan ketikkan pasal UUD 1945 berapa yang ingin Kamu cari :D\n\nAturan untuk menanyakan BAB dan Pasal pada UUD 1945:\n\n1. Pencarian Bab pada UUD 1945:\n- Gunakan angka Romawi untuk Bab I - XVI\n- Gunakan kata untuk Bab I, Bab V, dan Bab X\n- Contoh Pencarian: \"Bab II, bab ix, bab satu, bab lima, bab sepuluh\"\n\n2. Pencarian Pasal pada UUD 1945:\n- Gunakan angka desimal untuk Pasal 10 - 37\n- Gunakan kata untuk Pasal 1 - 9\n- Contoh Pencarian: \"Pasal 10, pasal 15, pasal satu, pasal enam\""
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