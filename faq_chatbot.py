import tkinter as tk
from tkinter import scrolledtext

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (first run only)
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")

# FAQs
faq_questions = [
    "What are your delivery charges?",
    "How long does delivery take?",
    "Can I return a product?",
    "What payment methods do you accept?",
    "Do you offer cash on delivery?",
    "How can I track my order?",
    "Do you ship internationally?",
    "Can I cancel my order?",
    "How do I contact customer support?",
    "What is your refund policy?"
]

faq_answers = [
    "Delivery charges are Rs. 200 nationwide.",
    "Delivery takes 3 to 5 working days.",
    "Yes, products can be returned within 7 days.",
    "We accept cash on delivery and bank transfer.",
    "Yes, cash on delivery is available.",
    "You can track your order using the tracking ID.",
    "No, we currently ship only within Pakistan.",
    "Orders can be cancelled before shipment.",
    "You can contact us via email or phone.",
    "Refunds are processed within 7 business days."
]

# Stopwords
stop_words = set(stopwords.words("english"))

# Preprocessing Function
def preprocess(text):
    text = text.lower()

    words = word_tokenize(text)

    filtered_words = [
        word for word in words
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(filtered_words)

# Preprocess FAQ Questions
processed_faqs = [
    preprocess(question)
    for question in faq_questions
]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    processed_faqs
)

# Chatbot Function
def get_response():

    user_question = user_input.get()

    if not user_question.strip():
        return

    processed_question = preprocess(
        user_question
    )

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    if best_score > 0.2:
        response = faq_answers[
            best_match_index
        ]
    else:
        response = (
            "Sorry, I do not understand your question."
        )

    chat_area.insert(
        tk.END,
        f"You: {user_question}\n"
    )

    chat_area.insert(
        tk.END,
        f"Bot: {response}\n\n"
    )

    user_input.delete(0, tk.END)

# Main Window
root = tk.Tk()

root.title("FAQ Chatbot")

root.geometry("700x500")

#chatbot label
tk.Label(
    root,
    text="FAQ Chatbot",
    font=("Arial", 16, "bold")
).pack(pady=(10,0))

# Chat Area
chat_area = scrolledtext.ScrolledText(
    root,
    width=80,
    height=20,
    font=("Arial", 11)
)

chat_area.pack( padx=10,pady=10 )

#input label
tk.Label(
    root,
    text="Type your question below:",
    font=("Arial", 12 , "bold")
).pack(pady=(0,5))

# User Input
user_input = tk.Entry(
    root,
    width=60,
    font=("Arial", 11)
)

user_input.pack(
    side=tk.LEFT,
    padx=10,
    pady=10
)

#using enter key to send message
user_input.bind("<Return>", lambda event: get_response())

# Send Button
send_button = tk.Button(
    root,
    text="Send",
    command=get_response,
    width=15
)

send_button.pack(
    side=tk.LEFT,
    padx=10
)

root.mainloop()