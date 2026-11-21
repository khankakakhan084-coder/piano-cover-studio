"""
token_manager.py
Smart Token Checker & Pool Manager for Hugging Face ZeroGPU.
- Manages Hugging Face Tokens for continuous free NVIDIA ZeroGPU compute.
- Loads tokens securely from storage/hf_tokens.txt or HF_TOKENS environment variable.
- Tracks ZeroGPU quota cooldowns and active states.
- Automatically selects the best fresh token with failover.
"""
import os
import json
import time
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger("piano_token_manager")
logger.setLevel(logging.INFO)

STATE_FILE = Path(__file__).resolve().parent / "storage" / "token_state.json"
TOKENS_FILE = Path(__file__).resolve().parent / "storage" / "hf_tokens.txt"


def load_tokens_from_disk_or_env() -> List[str]:
    """Loads HF tokens from storage/hf_tokens.txt or environment variables."""
    tokens = []
    # 1. From storage/hf_tokens.txt
    if TOKENS_FILE.exists():
        try:
            lines = TOKENS_FILE.read_text(encoding="utf-8").splitlines()
            for line in lines:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#"):
                    tokens.append(cleaned)
        except Exception as e:
            logger.warning(f"Error reading {TOKENS_FILE}: {e}")

    # 2. From environment variable HF_TOKENS (comma, space, or newline separated)
    env_tokens = os.environ.get("HF_TOKENS", "")
    if env_tokens:
        for t in env_tokens.replace(",", " ").split():
            t_clean = t.strip()
            if t_clean and t_clean not in tokens:
                tokens.append(t_clean)

    # 3. Single HF_TOKEN
    single_token = os.environ.get("HF_TOKEN", "").strip()
    if single_token and single_token not in tokens:
        tokens.append(single_token)

    return tokens


class TokenManager:
    """Manages Hugging Face token health, quota cooldowns, and automatic selection."""

    def __init__(self, tokens: Optional[List[str]] = None):
        self.tokens = list(tokens or load_tokens_from_disk_or_env())
        self.state_file = STATE_FILE
        self._ensure_storage()
        self.state = self._load_state()

    def _ensure_storage(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.state_file = Path("/tmp") / "piano_token_state.json"
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_state(self):
        try:
            self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not save token state: {e}")

    def reload_tokens(self):
        """Reloads tokens from disk/env."""
        self.tokens = load_tokens_from_disk_or_env()

    def verify_token_identity(self, token: str) -> Optional[dict]:
        """Fast check with HF API to get username and verify token validity."""
        try:
            r = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
                timeout=4.0,
            )
            if r.status_code == 200:
                data = r.json()
                return {
                    "valid": True,
                    "username": data.get("name", "unknown"),
                    "email": data.get("email", ""),
                    "is_pro": data.get("isPro", False),
                }
            elif r.status_code == 401:
                return {"valid": False, "error": "Invalid / Expired Token"}
        except Exception as ex:
            return {"valid": None, "error": str(ex)}
        return {"valid": False, "error": f"HTTP {r.status_code}"}

    def mark_exhausted(self, token: str, error_message: str):
        """Mark a token as having hit its ZeroGPU quota, cooldown for 60 mins."""
        now = time.time()
        cooldown = 3600  # 1 hour
        self.state[token] = {
            "status": "exhausted",
            "exhausted_at": now,
            "reset_at": now + cooldown,
            "last_error": error_message,
        }
        self._save_state()
        logger.warning(f"Token ...{token[-6:]} marked exhausted for 60 min. Error: {error_message}")

    def mark_used(self, token: str):
        """Mark a token as actively working."""
        now = time.time()
        entry = self.state.get(token, {})
        entry.update({
            "status": "active",
            "last_used": now,
            "success_count": entry.get("success_count", 0) + 1,
        })
        self.state[token] = entry
        self._save_state()

    def get_available_tokens(self) -> List[str]:
        """Returns ordered list of available tokens that are not on cooldown."""
        if not self.tokens:
            self.reload_tokens()

        now = time.time()
        available = []
        for token in self.tokens:
            info = self.state.get(token)
            if not info:
                available.append(token)
                continue
            if info.get("status") == "exhausted":
                reset_at = info.get("reset_at", 0)
                if now >= reset_at:
                    info["status"] = "active"
                    available.append(token)
                else:
                    continue
            else:
                available.append(token)
        return available or list(self.tokens)

    def get_pool_status(self) -> Dict[str, Any]:
        """Returns summary status of the token pool."""
        now = time.time()
        total = len(self.tokens)
        exhausted = 0
        active = 0
        details = []

        for idx, token in enumerate(self.tokens):
            info = self.state.get(token, {})
            is_exhausted = info.get("status") == "exhausted" and now < info.get("reset_at", 0)
            if is_exhausted:
                exhausted += 1
                rem_mins = int((info["reset_at"] - now) / 60)
                status_str = f"Cooldown ({rem_mins}m left)"
            else:
                active += 1
                status_str = "Ready"

            details.append({
                "index": idx + 1,
                "token_masked": f"...{token[-6:]}" if len(token) > 6 else "token",
                "status": status_str,
                "success_count": info.get("success_count", 0),
            })

        return {
            "total_tokens": total,
            "active_tokens": active,
            "cooling_tokens": exhausted,
            "tokens": details,
        }


token_manager = TokenManager()
