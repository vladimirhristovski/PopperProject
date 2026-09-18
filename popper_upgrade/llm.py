import os


def ensure_local_llm_api_key(port, api_key="EMPTY"):
    if port is not None:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
