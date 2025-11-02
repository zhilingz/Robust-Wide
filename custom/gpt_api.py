from openai import OpenAI
from dotenv import load_dotenv
import os

class UnifiedLLM:
    def __init__(self, provider: str, api_key: str, model: str):
        provider = provider.lower()
        base_urls = {
            "openai": "https://api.openai.com/v1",
            "doubao": "https://ark.cn-beijing.volces.com/api/v1",  # 豆包 Ark 平台
            "nano": "https://api.minimax.chat/v1",  # Nano / Minimax
            "gork": "https://api.gork.cn/v1",       # 假设 Gork 类似
        }
        if provider not in base_urls:
            raise ValueError(f"Unknown provider: {provider}")
        self.client = OpenAI(api_key=api_key, base_url=base_urls[provider])
        self.model = model

    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content


load_dotenv()

providers = {
    "openai": ("gpt-4o", os.getenv("OPENAI_API_KEY")),
    "doubao": ("doubao-lite-4k", os.getenv("DOUBAO_API_KEY")),
    "nano":   ("abab6.5-chat", os.getenv("NANO_API_KEY")),
    "gork":   ("gork-large", os.getenv("GORK_API_KEY")),
}

msg = [{"role": "user", "content": "请用一句话解释量子纠缠"}]

for name, (model, key) in providers.items():
    llm = UnifiedLLM(provider=name, api_key=key, model=model)
    reply = llm.chat(msg)
    print(f"[{name}] {reply}\n")
