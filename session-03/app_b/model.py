"""Everything that talks to the language model."""

import os

from dotenv import load_dotenv
from openai import OpenAI


def create_client():
    """Connect to OpenRouter using the key saved in .env."""
    load_dotenv()
    return OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def build_system_prompt(settings):
    """Turn the persona settings into the system prompt."""
    assistant = settings["assistant"]
    persona = assistant["persona"]

    prompt = f"You are {assistant['name']}, {persona['role']}. "
    prompt += f"Your tone is {persona['tone']}.\n"
    prompt += "Always follow these rules:\n"
    for rule in persona["rules"]:
        prompt += f"- {rule}\n"
    return prompt


def build_messages(settings, history):
    """Put the system prompt in front of the conversation so far."""
    system_message = {"role": "system",
                      "content": build_system_prompt(settings)}
    return [system_message] + history


def ask_model(client, settings, history):
    """Send the conversation to the model. Returns the reply and some stats."""
    try:
        messages = build_messages(settings, history)
        response = client.chat.completions.create(
            model=settings["model"]["name"],
            messages=messages,
            max_tokens=settings["model"].get("max_tokens", 200),
            temperature=settings["model"].get("temperature", 0.7),
        )
        return {
            "reply": response.choices[0].message.content,
            "messages_sent": len(messages),
            "tokens_in": response.usage.prompt_tokens,
            "tokens_out": response.usage.completion_tokens,
        }
    except Exception:
        # Something went wrong with the request; let the chat keep going.
        return None
