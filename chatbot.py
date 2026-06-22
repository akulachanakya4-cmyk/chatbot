import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# FAQs (Questions and Answers)
faq_data = {
    "What is Python?":
        "Python is a high-level programming language used for web development, AI, and data science.",

    "What is Artificial Intelligence?":
        "Artificial Intelligence is the simulation of human intelligence by machines.",

    "What is Machine Learning?":
        "Machine Learning is a branch of AI that enables computers to learn from data.",

    "Who developed Python?":
        "Python was developed by Guido van Rossum.",

    "What is NLP?":
        "Natural Language Processing is a field of AI that helps computers understand human language."
}

questions = list(faq_data.keys())

# Preprocess text
def preprocess(text):
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    stop_words = ENGLISH_STOP_WORDS
    words = [word for word in tokens if word not in stop_words]
    return " ".join(words)

processed_questions = [preprocess(q) for q in questions]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(processed_questions)

print("===== FAQ Chatbot =====")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(input_vector, vectors)
    index = similarity.argmax()
    score = similarity[0][index]

    if score > 0.2:
        print("Chatbot:", faq_data[questions[index]])
    else:
        print("Chatbot: Sorry, I don't know the answer to that question.")