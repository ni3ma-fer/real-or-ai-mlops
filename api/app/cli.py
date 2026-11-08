"""Gestion des utilisateurs et des clés, à lancer dans le conteneur de l'API :

    docker compose exec api python -m app.cli create-key alice
    docker compose exec api python -m app.cli create-key admin --admin
    docker compose exec api python -m app.cli list
    docker compose exec api python -m app.cli revoke roa_AbCdEfGh
"""

import argparse

from sqlalchemy import func, select

from .auth import generate_key, hash_key
from .db import ApiKey, SessionLocal, Usage, User, init_db


def create_key(name: str, admin: bool) -> None:
    with SessionLocal.begin() as s:
        user = s.scalar(select(User).where(User.name == name))
        if user is None:
            user = User(name=name, is_admin=admin)
            s.add(user)
            s.flush()
        elif admin and not user.is_admin:
            user.is_admin = True
        key = generate_key()
        s.add(ApiKey(user_id=user.id, key_hash=hash_key(key), prefix=key[:12]))
    print(f"Clé créée pour {name}{' (admin)' if admin else ''}. Elle ne sera plus jamais affichée :\n{key}")


def list_keys() -> None:
    with SessionLocal() as s:
        counts = dict(s.execute(select(Usage.user_id, func.count()).group_by(Usage.user_id)).all())
        rows = s.execute(select(User, ApiKey).join(ApiKey, ApiKey.user_id == User.id).order_by(User.name)).all()
    print(f"{'utilisateur':<15} {'admin':<6} {'clé':<14} {'active':<7} requêtes")
    for user, key in rows:
        print(f"{user.name:<15} {str(user.is_admin):<6} {key.prefix + '…':<14} {str(key.active):<7} {counts.get(user.id, 0)}")


def revoke(prefix: str) -> None:
    with SessionLocal.begin() as s:
        keys = s.scalars(select(ApiKey).where(ApiKey.prefix == prefix[:12])).all()
        for k in keys:
            k.active = False
    print(f"{len(keys)} clé(s) révoquée(s) (effectif sous 60 s).")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-key", help="crée l'utilisateur si besoin et lui donne une nouvelle clé")
    c.add_argument("name")
    c.add_argument("--admin", action="store_true")
    sub.add_parser("list", help="liste les clés et le nombre de requêtes par utilisateur")
    r = sub.add_parser("revoke", help="désactive une clé à partir de son début (roa_xxxxxxxx)")
    r.add_argument("prefix")
    args = p.parse_args()

    init_db()
    if args.cmd == "create-key":
        create_key(args.name, args.admin)
    elif args.cmd == "list":
        list_keys()
    else:
        revoke(args.prefix)


if __name__ == "__main__":
    main()
