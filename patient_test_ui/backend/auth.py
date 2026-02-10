"""Simple user authentication and storage for the Patient Agent UI."""

from __future__ import annotations

import json
import os
import secrets
import hashlib
import hmac
from pathlib import Path
from typing import Dict, List


class UserStore:
    """Persisted user credential store with salted SHA-256 hashes."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._users: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        self._users = data
            except json.JSONDecodeError:
                # Fallback to empty store on corruption
                self._users = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._users, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return hashed

    def list_users(self) -> List[str]:
        return sorted(self._users.keys())

    def register(self, username: str, password: str) -> None:
        username = (username or "").strip()
        if not username:
            raise ValueError("用户名不能为空")
        if not password:
            raise ValueError("密码不能为空")
        if username in self._users:
            raise ValueError("该用户名已存在")

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        self._users[username] = {
            "salt": salt,
            "password_hash": password_hash,
        }
        self._save()

    def authenticate(self, username: str, password: str) -> bool:
        entry = self._users.get(username)
        if not entry or "salt" not in entry or "password_hash" not in entry:
            return False
        candidate = self._hash_password(password, entry["salt"])
        return hmac.compare_digest(candidate, entry["password_hash"])

