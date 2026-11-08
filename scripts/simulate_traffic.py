"""Simule du trafic utilisateur pour la démo de drift.

    python scripts/simulate_traffic.py --scenario normal --n 300
    python scripts/simulate_traffic.py --scenario drift  --n 300   # arrivée d'IA inconnues

Scénarios (images de data/samples/) :
  normal : 50 % vraies photos, 50 % images des 5 générateurs vus à l'entraînement
  drift  : 30 % vraies photos, 70 % Midjourney + Wukong (jamais vus par le modèle v1)
Clé : --key ou variable API_KEY (défaut : ALICE_KEY de loadtest/.keys.env).
"""

import argparse
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
SCENARIOS = {
    "normal": (0.5, ["ADM", "BigGAN", "GLIDE", "SD15", "VQDM"]),
    "drift": (0.3, ["Midjourney", "Wukong"]),
}


def default_key(keys_file: str) -> str:
    for line in (ROOT / keys_file).read_text().splitlines():
        if line.startswith("ALICE_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("Pas de clé : --key ou API_KEY")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=SCENARIOS, default="normal")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--key", default=os.getenv("API_KEY"))
    p.add_argument("--keys-file", default="loadtest/.keys.env")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    key = args.key or default_key(args.keys_file)

    real_share, gens = SCENARIOS[args.scenario]
    real = sorted((SAMPLES / "Real").iterdir())
    fake = [f for g in gens for f in sorted((SAMPLES / g).iterdir())]
    rng = random.Random()
    # Tirage SANS remise (tant que possible) : pas d'image envoyée deux fois
    n_real = sum(rng.random() < real_share for _ in range(args.n))
    pick = lambda pool, k: rng.sample(pool, k) if k <= len(pool) else [rng.choice(pool) for _ in range(k)]
    files = pick(real, n_real) + pick(fake, args.n - n_real)
    rng.shuffle(files)

    session = requests.Session()
    session.headers["X-API-Key"] = key

    def send(f: Path):
        r = session.post(f"{args.url}/predict", files={"file": (f.name, f.read_bytes())}, timeout=60)
        truth = "real" if f.parent.name == "Real" else "ai"
        return f.parent.name, r.status_code, r.json() if r.ok else None, truth

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(send, files))
    ok = [r for r in results if r[1] == 200]
    correct = Counter(g for g, _, res, truth in ok if res["label"] == truth)
    total = Counter(g for g, *_ in ok)
    versions = Counter(res["model_version"] for _, _, res, _ in ok)
    print(f"Scénario {args.scenario} : {len(ok)}/{len(results)} réponses OK en {time.time() - t0:.0f} s")
    print("  versions du modèle :", dict(versions))
    for g in sorted(total):
        print(f"  {g:<11} {correct[g]}/{total[g]} correctes ({correct[g] / total[g]:.0%})")
    print(f"  global : {sum(correct.values()) / max(1, len(ok)):.1%} de bonnes réponses")


if __name__ == "__main__":
    main()
