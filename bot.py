import os
import sys
import time
import requests
import google.generativeai as genai

# --- CONFIGURATION & ENV VARS ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    """Utility function to send status/error alerts directly to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


# --- AI PROVIDER IMPLEMENTATIONS ---

def call_gemini(prompt: str) -> str:
    """Uses Gemini 2.0 Flash / 1.5 Flash (active, high-quota models)."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Active Gemini models in order of preference
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]
    
    for model_name in models_to_try:
        max_retries = 3
        backoff = 2
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response.text:
                    return response.text
            except Exception as e:
                err_str = str(e).lower()
                # Check for rate limit / quota exhaustion
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    if attempt < max_retries - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RuntimeError(f"Rate limit exceeded on Gemini ({model_name}).")
                # Check for invalid/deprecated model error
                elif "not found" in err_str or "invalid" in err_str:
                    print(f"Gemini model {model_name} invalid or unavailable, trying fallback model...")
                    break  # Break inner loop to try next model in models_to_try
                else:
                    raise RuntimeError(f"Gemini execution error ({model_name}): {e}")
    
    raise RuntimeError("All configured Gemini models failed.")


def call_groq(prompt: str) -> str:
    """Uses Groq REST API with active Llama models."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Active Groq models
    models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        max_retries = 3
        backoff = 2
        
        for attempt in range(max_retries):
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=30)
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"]
                
                # Rate limit status
                if res.status_code == 429:
                    if attempt < max_retries - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RuntimeError(f"Rate limit (429) hit on Groq ({model_name}).")
                
                # Model or auth errors
                err_msg = res.json().get("error", {}).get("message", res.text)
                if res.status_code == 404 or "model" in err_msg.lower():
                    print(f"Groq model {model_name} not available, trying fallback...")
                    break
                else:
                    raise RuntimeError(f"Groq API error ({res.status_code}): {err_msg}")
            
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Groq connection error: {e}")

    raise RuntimeError("All configured Groq models failed.")


def call_nvidia(prompt: str) -> str:
    """Uses NVIDIA NIM API with active Llama 3.3 model."""
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY is not configured.")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    model_name = "meta/llama-3.3-70b-instruct"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 1024
    }

    max_retries = 3
    backoff = 2

    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            
            if res.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError("Rate limit (429) hit on NVIDIA NIM API.")
            
            err_msg = res.json().get("error", {}).get("message", res.text)
            if "model" in err_msg.lower() or res.status_code == 404:
                raise RuntimeError(f"Invalid model error from NVIDIA: {err_msg}")
            else:
                raise RuntimeError(f"NVIDIA API error ({res.status_code}): {err_msg}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"NVIDIA connection error: {e}")

    raise RuntimeError("NVIDIA API failed after retries.")


# --- MAIN ORCHESTRATION ---

def main():
    prompt = "Summarize the top AI news stories from today into a concise 3-bullet-point executive digest."
    
    providers = [
        ("Gemini", call_gemini),
        ("Groq", call_groq),
        ("NVIDIA", call_nvidia)
    ]
    
    error_logs = []

    for name, provider_func in providers:
        print(f"Attempting response generation using {name}...")
        try:
            output = provider_func(prompt)
            print(f"Success via {name}!")
            send_telegram_message(f"🤖 *Daily AI News Digest* (via {name})\n\n{output}")
            return
        except Exception as e:
            error_msg = f"❌ [{name}] {str(e)}"
            print(error_msg)
            error_logs.append(error_msg)

    # If all providers fail, compile detailed diagnostic error message to Telegram
    failure_report = (
        "⚠️ *All AI Providers Failed*\n\n"
        "Detailed error diagnostics:\n" + "\n".join(error_logs)
    )
    send_telegram_message(failure_report)
    sys.exit(1)

if __name__ == "__main__":
    main()
