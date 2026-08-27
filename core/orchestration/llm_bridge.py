from __future__ import annotations

import os
import socket
import time
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

# Explicitly load .env relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(BASE_DIR, ".env")

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except ImportError:
    pass

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class LlamaCppEngine:
    """
    Offline local LLM Engine using llama-cpp-python.
    """
    def __init__(
        self,
        model_filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx: int = 2048,
        n_threads: int = 4,
    ):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed.")

        model_path = os.path.join(BASE_DIR, "models", model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. Place the GGUF file in models/"
            )

        print(f"[JARVIS LLM] Loading offline model from {model_path} ...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        print("[JARVIS LLM] Offline model loaded.")

    def generate(self, system_prompt: str, user_input: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()


class GroqEngine:
    """
    Groq API Engine with Auto-Model Sanitization, Multi-Key Rotation, and Detailed Debug Logging.
    """

    VALID_MODELS = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "groq/compound",
        "groq/compound-mini",
        "allam-2-7b",
    ]

    def __init__(
        self,
        api_keys: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 20.0,
    ):
        if requests is None:
            raise ImportError("The 'requests' package is not installed (pip install requests).")

        raw_keys = api_keys or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or ""
        self.api_keys = [k.strip() for k in raw_keys.replace(" ", "").split(",") if k.strip()]

        if not self.api_keys:
            raise RuntimeError("No Groq API key (gsk_...) found in .env file.")

        # Default model updated to 'openai/gpt-oss-120b'
        target_model = model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
        if target_model not in self.VALID_MODELS:
            target_model = "openai/gpt-oss-120b"
        
        self.model = target_model
        self.base_url = "https://api.groq.com/openai/v1"
        self.timeout = timeout
        self._current_index = 0

    def generate(self, system_prompt: str, user_input: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        total_keys = len(self.api_keys)
        last_error = None

        for attempt in range(total_keys):
            key_index = self._current_index
            self._current_index = (self._current_index + 1) % total_keys

            active_key = self.api_keys[key_index]
            key_num = key_index + 1

            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json",
            }

            try:
                print(f"[JARVIS LLM] Groq Key #{key_num}/{total_keys} | Model: {self.model}")
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

                # Catch 404 (Invalid/Unknown key), 401, 403, and 429
                if resp.status_code in (404, 401, 403, 429):
                    print(f"[JARVIS LLM] Key #{key_num} failed HTTP {resp.status_code}: {resp.text.strip()}")
                    print(f"[JARVIS LLM] Switching key...")
                    last_error = f"HTTP {resp.status_code} ({resp.text.strip()})"
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

            except Exception as exc:
                print(f"[JARVIS LLM] Key #{key_num} error: {exc}. Trying next key...")
                last_error = exc

        raise RuntimeError(f"All {total_keys} Groq API keys failed. Last error: {last_error}")


class HybridLLMBridge:
    CONNECTIVITY_CHECK_INTERVAL_SECONDS = 15
    CONNECTIVITY_TEST_HOST = "8.8.8.8"
    CONNECTIVITY_TEST_PORT = 53
    CONNECTIVITY_TIMEOUT = 1.5

    def __init__(
        self,
        model_filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx: int = 2048,
        n_threads: int = 4,
        force_mode: Optional[str] = None,
    ):
        self._model_filename = model_filename
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._force_mode = force_mode

        self._groq_engine: Optional[GroqEngine] = None
        self._local_engine: Optional[LlamaCppEngine] = None

        self._last_check_time = 0.0
        self._last_online_result = False

    def _is_online(self) -> bool:
        if self._force_mode == "online":
            return True
        if self._force_mode == "offline":
            return False

        now = time.time()
        if (now - self._last_check_time) < self.CONNECTIVITY_CHECK_INTERVAL_SECONDS:
            return self._last_online_result

        online = False
        try:
            socket.setdefaulttimeout(self.CONNECTIVITY_TIMEOUT)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.CONNECTIVITY_TEST_HOST, self.CONNECTIVITY_TEST_PORT))
            s.close()
            online = True
        except OSError:
            online = False

        self._last_check_time = now
        self._last_online_result = online
        return online

    def _get_groq(self) -> GroqEngine:
        if self._groq_engine is None:
            self._groq_engine = GroqEngine()
        return self._groq_engine

    def _get_local(self) -> LlamaCppEngine:
        if self._local_engine is None:
            self._local_engine = LlamaCppEngine(
                model_filename=self._model_filename,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
            )
        return self._local_engine

    def generate_response(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        online = self._is_online()
        have_groq_key = bool(os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY"))

        if online and have_groq_key:
            try:
                engine = self._get_groq()
                return engine.generate(
                    system_prompt=system_prompt,
                    user_input=user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                print(f"[JARVIS LLM] Groq API failed ({exc}), falling back to offline model.")

        try:
            engine = self._get_local()
            return engine.generate(
                system_prompt=system_prompt,
                user_input=user_input,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            return f"[Model Generation Error: {exc}]"


# Backward compatibility alias for cli.py and brain.py
LlamaCppBridge = HybridLLMBridge
