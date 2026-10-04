import os
import requests
from duckduckgo_search import DDGS
from groq import Groq

# --- 1. CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- 2. FETCH REAL-TIME AI NEWS ---
def fetch_ai_news() -> str:
    print("Fetching AI news from DuckDuckGo...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news("artificial intelligence tech news", timelimit="d", max_results=5))
            articles = [f"- {item['title']}: {item['body']}" for item in results]
            return "\n".join(articles)
    except Exception as e:
        print(f"Error fetching news: {e}")
        return "Artificial Intelligence major updates in LLMs and hardware."

# --- 3. MULTI-MODEL FALLBACK ENGINE ---
def summarize_news(news_text: str) -> str:
    prompt = (
        "You are an expert AI news curator. Summarize the following news updates into "
        "a clean, engaging Telegram update using bold headers and bullet points:\n\n"
        f"{news_text}"
    )

    # Provider 1: Groq (Llama 3.3 70B)
    try:
        print("Attempting Groq (Llama 3.3 70B)...")
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq failed ({e}). Falling back to NVIDIA NIM...")

    # Provider 2: NVIDIA NIM (Llama 3 70B)
    try:
        print("Attempting NVIDIA NIM...")
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "meta/llama-3.1-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        res = requests.post(url, json=payload, headers=headers).json()
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"NVIDIA NIM failed ({e}). Falling back to Google Gemini...")

    # Provider 3: Google Gemini
    try:
        print("Attempting Google Gemini...")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        res = requests.post(url, json=payload).json()
        return res["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Gemini failed ({e}).")
        return "⚠️ All AI providers failed due to rate limits."

# --- 4. SEND TO TELEGRAM ---
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)

if __name__ == "__main__":
    raw_news = fetch_ai_news()
    summary = summarize_news(raw_news)
    send_telegram(f"🤖 *Daily AI Briefing*\n\n{summary}")