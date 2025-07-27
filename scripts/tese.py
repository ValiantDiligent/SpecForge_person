import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print_highlight(response.json())