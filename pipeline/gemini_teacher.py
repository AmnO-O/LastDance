"""Gemini text backend for simulator-verified training demonstrations."""
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class GeminiTeacher:
    def __init__(self, keys, model):
        self.keys = list(dict.fromkeys(key.strip() for key in keys if key.strip()))
        if not self.keys or not re.fullmatch(r"gemini-[a-zA-Z0-9.-]+", model):
            raise ValueError("Configure Gemini keys and a valid Gemini model name")
        self.model = model
        self.index = 0
        self.disabled = set()

    def complete(self, messages, settings):
        payload = {
            "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1024,
                                 "responseMimeType": "application/json"},
        }
        last = "no active key"
        for attempt in range(len(self.keys)):
            index = self.index
            self.index = (index + 1) % len(self.keys)
            if index in self.disabled:
                continue
            request = Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                data=json.dumps(payload).encode(),
                headers={"x-goog-api-key": self.keys[index], "Content-Type": "application/json"},
            )
            try:
                with urlopen(request, timeout=45) as response:
                    body = json.load(response)
            except HTTPError as exc:
                last = f"HTTP {exc.code}"
                invalid_key = exc.code == 400 and b"API_KEY_INVALID" in exc.read()
                if invalid_key or exc.code in (401, 403):
                    self.disabled.add(index)
                elif exc.code == 429 or 500 <= exc.code <= 599:
                    time.sleep(min(2 ** attempt, 8))
                else:
                    raise RuntimeError(f"Gemini request failed ({last})") from None
                continue
            except (URLError, TimeoutError):
                last = "network failure"
                time.sleep(min(2 ** attempt, 8))
                continue
            candidates = body.get("candidates") or []
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            return "".join(part.get("text", "") for part in parts if not part.get("thought"))
        raise RuntimeError(f"Gemini keys exhausted ({last})")
