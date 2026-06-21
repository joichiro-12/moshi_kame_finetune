from openai import OpenAI

MODEL = "llm-jp/llm-jp-4-8b-thinking"

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM はキー検証しないが空文字だと弾かれることがある
)


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": "あなたは日本語で自然に会話するアシスタントです。あなたの回答は音声に変換されるので、markdown記法や発音に関係のない記号（「\" ー ;」など）は含めないこと。",
        }
    ]

    print("チャットを開始します。終了するには exit または quit を入力してください。")

    while True:
        try:
            user_input = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("終了します。")
            break

        messages.append({"role": "user", "content": user_input})

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
            extra_body={"reasoning_effort": "low"},  # KAME 用途では low 推奨
        )

        assistant_msg = resp.choices[0].message
        assistant_content = assistant_msg.content or ""
        messages.append({"role": "assistant", "content": assistant_content})

        print(f"\nAssistant> {assistant_content}")


if __name__ == "__main__":
    main()
