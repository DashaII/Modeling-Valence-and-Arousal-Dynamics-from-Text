import requests
import json
import configs

def print_available_models():
    url = configs.ollama_url + "api/tags"
    response = requests.get(url)
    if response.status_code == 200:
        models = response.json()
        print("Available models:")
        for model in models.get("models", []):
            print("-", model["name"])
    else:
        print("Error:", response.status_code, response.text)


def ask_ollama(prompt, model_name="qwen2.5:72b"):
    url = configs.ollama_url + "v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama"
    }
    data = {
        "model": model_name,
        "prompt": prompt
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check for successful response
    res = ""
    if response.status_code == 200:
        result = response.json()  # Parse JSON into Python dict
        # Pretty-print the entire JSON
        # print(json.dumps(result, indent=2))

        # Extract and print just the generated text
        if "choices" in result and len(result["choices"]) > 0:
            res = result["choices"][0]["text"].strip()
        else:
            print("No choices found in the response.")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

    return res


def ask_ollama_chat(prompt, model_name="gpt-oss:120b", temperature=0.1):
    url = configs.ollama_url + "api/chat"
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model_name,
        "messages": [
            {
                "role": "assistant",
                "content": prompt
            },
            {
                "role": "user",
                "content": "Produce the labels now."
            }
        ],
        "stream": False,
        "temperture": temperature,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        return result["message"]["content"].strip()
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return ""
