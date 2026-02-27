import openai
import configs

def ask_openai(model=configs.model_openai_gpt_5_nano, instructions=None, input=None):
    openai.api_key = configs.openai_key
    client = openai.OpenAI(api_key=configs.openai_key)

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input = input
        # input="Write a one-sentence bedtime story about a unicorn."
    )

    return response.output_text


def ask_openai_chat(model_name=configs.model_openai_gpt_5_nano, prompt=None):
    openai.api_key = configs.openai_key
    client = openai.OpenAI(api_key=configs.openai_key)

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "assistant",
                "content": prompt
            },
            {
                "role": "user",
                "content": "Produce the labels now."
            }
        ]
    )
    return response.output_text


