"""Vérification de la clé API envoyée dans l'en-tête X-API-Key."""

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import select

from .db import ApiKey, SessionLocal, User

KEY_PREFIX = "roa_"  # « real-or-ai », pour reconnaître nos clés (comme ghp_ chez GitHub)
CACHE_SECONDS = 60  # évite une requête SQL par appel ; une clé révoquée l'est au plus tard après 60 s

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Caller:
    user_id: int
    name: str
    is_admin: bool


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


_cache: dict[str, tuple[float, Caller | None]] = {}
_lock = threading.Lock()


def _lookup(key_hash: str) -> Caller | None:
    with _lock:
        hit = _cache.get(key_hash)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    with SessionLocal() as s:
        row = s.execute(
            select(User.id, User.name, User.is_admin)
            .join(ApiKey, ApiKey.user_id == User.id)
            .where(ApiKey.key_hash == key_hash, ApiKey.active.is_(True))
        ).first()
    caller = Caller(*row) if row else None
    with _lock:
        _cache[key_hash] = (time.monotonic() + CACHE_SECONDS, caller)
    return caller


def require_key(key: str | None = Security(api_key_header)) -> Caller:
    if not key:
        raise HTTPException(401, "Clé API manquante : ajoutez l'en-tête X-API-Key", headers={"WWW-Authenticate": "ApiKey"})
    caller = _lookup(hash_key(key))
    if caller is None:
        raise HTTPException(401, "Clé API invalide ou révoquée", headers={"WWW-Authenticate": "ApiKey"})
    return caller


def require_admin(caller: Caller = Depends(require_key)) -> Caller:
    if not caller.is_admin:
        raise HTTPException(403, "Réservé aux administrateurs")
    return caller
