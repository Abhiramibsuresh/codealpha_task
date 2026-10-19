import string
from flask import Flask, render_template, request, jsonify
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# FAQ dataset
faqs = [
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking link sent to your registered email."
    },
    {
        "question": "What is your return policy?",
        "answer": "You can return eligible products within 7 days of delivery."
    },
    {
        "question": "How long does delivery take?",
        "answer": "Delivery usually takes 3 to 5 business days."
    },
    {
        "question": "Which payment methods do you accept?",
        "answer": "We accept UPI, debit cards, credit cards, and net banking."
    },
    {
        "question": "How can I cancel my order?",
        "answer": "You can cancel your order before it is shipped by visiting your orders page."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact customer support through the Help section on our website."
    },
    {
        "question": "Can I get a refund?",
        "answer": "Refunds are processed after the returned product is approved, according to the refund policy."
    },
    {
        "question": "Do you offer free shipping?",
        "answer": "Free shipping is available on eligible orders. Check the shipping details at checkout."
    },
    {
        "question": "Can I change my delivery address?",
        "answer": "You may be able to change your delivery address before your order is shipped."
    },
    {
        "question": "How do I create an account?",
        "answer": "Click the Sign Up button on our website and enter the required details."
    }
]


# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    tokens = wordpunct_tokenize(text)
    return " ".join(tokens)


# Prepare FAQ vectors once when the app starts
questions = [
    preprocess_text(faq["question"])
    for faq in faqs
]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)


# Chatbot response function
def get_answer(user_question):
    processed_question = preprocess_text(user_question)

    if not processed_question.strip():
        return "Please enter a question."

    greetings = [
        "hi", "hello", "hey",
        "good morning", "good evening"
    ]

    if processed_question in greetings:
        return "Hello! Welcome to our store. How can I help you today?"

    if processed_question in [
        "thanks", "thank you", "thank you very much"
    ]:
        return "You're welcome! Happy shopping!"

    if processed_question in [
        "how are you",
        "how are you doing"
    ]:
        return "I'm doing great! 😊 How can I help you with your shopping today?"

    if processed_question in [
            "bye",
            "goodbye",
            "tata"
    ]:
        return "Bye! 😊 Have a great day!"

    if "your name" in processed_question or "who are you" in processed_question:
        return "I ShopMate,Your friendly online shoping assistant!😊"
    
    
    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarities = cosine_similarity(
        user_vector, faq_vectors
    )[0]

    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    if best_score < 0.30:
        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please contact customer support."
        )

    return faqs[best_match_index]["answer"]


# Display the website
@app.route("/")
def home():
    return render_template("index.html")


# Receive questions from the website
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("message", "")

    if not isinstance(question, str):
        return jsonify({"answer": "Please enter a valid question."}), 400

    answer = get_answer(question)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)

