"""
Duka Fresh Support Chatbot — core logic.

This is the same retrieval-based approach from the original Colab notebook
(TF-IDF vectorization + cosine similarity), just moved into a standalone
module so it can be imported by a web server instead of only running
inside a notebook.
"""

import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "customer_support_dataset.csv")


def load_dataset(path=DATASET_PATH):
    """
    Reads the CSV dataset and returns a list of dicts:
    [{"intent": ..., "category": ..., "question": ..., "answer": ...}, ...]
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_questions_and_answers(rows):
    """
    Splits the dataset into parallel lists:
    questions (used to train the retrieval model)
    answers   (the response returned for a matched question)
    intents   (for reporting / evaluation)
    """
    questions = [r["question"] for r in rows]
    answers = [r["answer"] for r in rows]
    intents = [r["intent"] for r in rows]
    return questions, answers, intents


class SupportChatbot:
    def __init__(self, dataset_path=DATASET_PATH, threshold=0.25):
        self.rows = load_dataset(dataset_path)
        self.questions, self.answers, self.intents = get_questions_and_answers(self.rows)

        # threshold: minimum similarity score to accept a match.
        # Below this, the bot admits it doesn't understand rather than
        # guessing (important for the "test and improve" step).
        self.threshold = threshold

        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_message):
        """
        Returns (answer, matched_question, similarity_score, intent)
        """
        user_vec = self.vectorizer.transform([user_message])
        similarities = cosine_similarity(user_vec, self.question_vectors)[0]

        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.threshold:
            return (
                "Sorry, I didn't quite understand that. Could you rephrase, "
                "or type 'agent' to talk to a human?",
                None,
                float(best_score),
                "unknown",
            )

        return (
            self.answers[best_idx],
            self.questions[best_idx],
            float(best_score),
            self.intents[best_idx],
        )

    def chat(self):
        """Terminal-style loop — still works if you want to test locally without the web UI."""
        print("Duka Fresh Support Bot — type 'exit' to quit\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Bot: Asante for chatting with Duka Fresh! Kwaheri \U0001F44B")
                break
            answer, matched_q, score, intent = self.get_response(user_input)
            print(f"Bot: {answer}")
