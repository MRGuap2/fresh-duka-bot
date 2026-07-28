"""
Duka Fresh Support Chatbot — web server.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000 in your browser.

Deploy it so others can reach it outside your machine — see README.md
for free hosting options (Render, Railway, Hugging Face Spaces).
"""

import os

from flask import Flask, jsonify, render_template, request

from chatbot import SupportChatbot

app = Flask(__name__)
bot = SupportChatbot()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    answer, matched_question, score, intent = bot.get_response(user_message)

    return jsonify(
        {
            "answer": answer,
            "matched_question": matched_question,
            "score": round(score, 3),
            "intent": intent,
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "questions_loaded": len(bot.questions)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False and host 0.0.0.0 so it also works fine once deployed
    app.run(host="0.0.0.0", port=port, debug=False)
