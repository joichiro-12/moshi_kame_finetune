from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="llm-jp/llm-jp-4-8b-thinking",
    messages=[
        {"role": "user", "content": "夕食におすすめの和食を一品、短く"},
    ],
    max_tokens=128,
    extra_body={"reasoning_effort": "low"},
)

print(response.model_dump_json(indent=2))
