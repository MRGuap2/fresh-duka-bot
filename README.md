# Duka Fresh Support Chatbot — Web App

This turns your Colab notebook's TF-IDF chatbot into a small website you (and
customers) can chat with in a browser — no notebook required.

## What's in here

```
duka_fresh_bot/
├── app.py                          # Flask server (the "always running" part)
├── chatbot.py                      # Your original SupportChatbot class
├── customer_support_dataset.csv    # Starter dataset — REPLACE with your full 150-row CSV
├── templates/index.html            # The chat page people will see
├── requirements.txt
└── README.md
```

**Important:** the `customer_support_dataset.csv` here only has ~20 sample
rows (the ones visible in your notebook's output). Replace this file with
your real, full dataset before sharing the link with real customers — just
keep the same filename and the same 4 columns (`intent,category,question,answer`).

## 1. Run it on your own computer first

```bash
cd duka_fresh_bot
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser and chat with it. This
already satisfies "talk to it outside the notebook" — but it only works on
your machine.

## 2. Put it on the internet so anyone can chat with it

To get a real public link, you need to host it somewhere that keeps a server
running 24/7. Free options that work well for a small project like this:

### Option A — Render (easiest, free tier)
1. Push this folder to a GitHub repo.
2. Go to https://render.com → New → Web Service → connect your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy — Render gives you a public URL like `https://duka-fresh-bot.onrender.com`.

### Option B — Hugging Face Spaces (free, good for demos)
1. Create a new Space at https://huggingface.co/new-space, choose "Docker" or
   "Gradio/Streamlit → Blank" with Python.
2. Upload these files.
3. Spaces gives you a public URL automatically.

### Option C — Railway (free trial credits)
Similar flow to Render: connect GitHub repo, set start command to
`gunicorn app:app`, deploy.

Any of these gives you a shareable link you can post on WhatsApp/Instagram
bio so customers can message the bot directly.

## 3. Swapping in your real dataset

Just replace `customer_support_dataset.csv` with your full file (same column
names) and restart the app — no other code changes needed.

## 4. Extending it later

Some natural next steps once this is live:
- Hook it into an actual WhatsApp Business number using the WhatsApp Cloud API,
  so customers chat with it where they already are.
- Log unmatched questions (where `intent == "unknown"`) to a file so you can
  see what people are asking that isn't in your dataset yet, and add more rows.
