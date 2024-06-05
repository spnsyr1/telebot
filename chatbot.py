import logging
import json
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
        responses.append(intent["responses"][0])  # Memilih respon pertama

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

prev_answer = ""


# Method to predict answer based on user input
def predict_answer(user_input):
    global prev_answer
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    predicted_answer = classifier.predict(user_input_tfidf)[0]
    return predicted_answer


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Halo! Selamat datang di Bot Pengenal UUD 1945. Silahkan ketikkan pasal UUD 1945 berapa yang ingin anda cari :D"
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global prev_answer
    text = update.message.text
    # text = text.lower()
    # kalo mau edit inputan user di sini
    # text = inputan user

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