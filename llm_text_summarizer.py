import requests

# ✅ Together AI API Key 
API_KEY = "d651d3ae6d15225cd4ffaae91038cf1b16ee7a480143c07286d8d0b7210a51fc"

API_URL = "https://api.together.xyz/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json" 
}

def gpt_summarize_medical_report(text: str) -> str:
    try:
        body = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # ✅ Fast, free model
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Summarize clinical reports clearly,"
                    " focusing on diagnosis, tests, and treatment."
                },
                {
                    "role": "user",
                    "content": f"Summarize this medical report:\n\n{text}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }

        response = requests.post(API_URL, headers=HEADERS, json=body)

        if response.status_code != 200:
            print("STATUS:", response.status_code)
            print("RESPONSE TEXT:", response.text)
            return f"API Error {response.status_code}: {response.text}"

        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    except Exception as e:
        return f"Unexpected error: {e}"
    
    
