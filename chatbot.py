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

# Common English filler words, plus common Swahili/Sheng filler words that show up
# across many intents (greetings, connectors, politeness words). Filtering these out
# stops the vectorizer from matching messages on shared sentence structure
# ("What's your ___?", "Naeza pata ___?") instead of the actual topic word.
ENGLISH_STOP_WORDS = [
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "what's", "whats", "what", "how", "do", "does", "did", "can", "could", "will",
    "would", "should", "to", "of", "in", "on", "at", "for", "with", "and", "or",
    "please", "just", "so", "if", "have", "has", "had", "get", "got",
]

SWAHILI_SHENG_STOP_WORDS = [
    "naeza", "naomba", "nataka", "nako", "wewe", "mimi", "sisi", "yako", "yangu",
    "yenu", "kwa", "na", "ni", "hii", "hiyo", "hizo", "je", "basi",
    "tafadhali", "asante",
    "samahani", "kuuliza", "kupata", "kuwa", "gani", "lini", "ngapi",
    "bado", "tu", "kidogo", "sana", "msee", "boss", "excuse",
]
# Note: greeting words like "mambo", "vipi", "sasa", "habari", "poa", "niaje" are
# deliberately NOT stop-worded — for several short training questions (e.g. "Mambo
# vipi", "Habari", "Sasa") these words are the entire content of the message, so
# filtering them out would leave nothing for the vectorizer to match on. Likewise
# "pata" and "wapi" are kept since they carry real meaning in "iko wapi" (where is
# it) style questions used across order/location intents.

STOP_WORDS = ENGLISH_STOP_WORDS + SWAHILI_SHENG_STOP_WORDS


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

        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=STOP_WORDS)
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

