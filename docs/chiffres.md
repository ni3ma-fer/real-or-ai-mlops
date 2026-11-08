# Chiffres pour la présentation

Tous les chiffres mesurés au fil du projet, avec leur contexte. Machine : PC portable, Intel i7-8665U (4 cœurs / 8 threads), 32 Go RAM, sans GPU utilisable (MX250 2 Go non prise en charge par PyTorch).

## Données — Tiny-GenImage (mesuré le 2026-11-08)

| | Train | Validation |
|---|---|---|
| Images réelles | 14 000 | 3 500 |
| Images IA | 14 000 (2 000 × 7 générateurs) | 3 500 (500 × 7) |
| Total | 28 000 | 7 000 |

- Taille téléchargée : 8,36 Go (18 fichiers parquet), téléchargement ≈ 10 min.
- Le dataset annonce 8 générateurs mais **SD 1.4 est absent** : 7 présents (ADM, BigGAN, GLIDE, Midjourney, SD 1.5, VQDM, Wukong).
- **Biais de format** : 100 % des vraies photos sont en JPEG (tailles variées), 100 % des images IA en PNG carrés (128 px BigGAN → 1024 px Midjourney). Correctif : toutes les images sont réencodées en JPEG qualité 90 avant le modèle, à l'entraînement comme dans l'API.
- Découpage pour la démo de drift : **Midjourney et Wukong exclus** de l'entraînement du modèle v1 → v1 apprend sur 14 000 réelles + 10 000 IA (5 générateurs).

## Modèle v1

- Encodeur : CLIP ViT-B/32 (OpenAI), vecteur de 512 nombres par image ; tête : régression logistique (méthode UnivFD, CVPR 2023).
- Calcul des embeddings sur CPU : ≈ 20 images/s.
- Temps de calcul des embeddings : train 28 000 images en 24,2 min, validation 7 000 en 6,0 min → **≈ 30 min au total** (19,5 images/s).
- Entraînement du classifieur : **0,3 s** (24 000 vecteurs).

### Résultats v1 (validation, seuil 0,5)

| Catégorie | Taux de bonnes réponses v1 | Si entraîné sur tout (comparaison) |
|---|---|---|
| Vraies photos | 92,7 % | 90,8 % |
| ADM | 97,4 % | 96,6 % |
| BigGAN | 98,6 % | 98,2 % |
| GLIDE | 99,0 % | 99,2 % |
| SD 1.5 | 78,4 % | 91,0 % |
| VQDM | 81,6 % | 78,2 % |
| **Midjourney (jamais vu par v1)** | **77,6 %** | 92,0 % |
| **Wukong (jamais vu par v1)** | **57,8 %** | 86,0 % |

- v1 sur les générateurs connus : **accuracy 92,0 %, AUC 0,979**.
- v1 sur toute la validation (connus + inconnus) : ≈ 88,5 %. Le modèle entraîné sur tout fait 91,2 %.
- **Message clé** : face à un générateur jamais vu, la performance s'effondre (Wukong : 57,8 %, à peine mieux que le hasard). C'est exactement le problème que la détection de drift doit repérer.
- Effet de bord : apprendre Wukong (basé sur Stable Diffusion) fait aussi monter SD 1.5 de 78 % à 91 %.

### Modèle exporté

- Fichier ONNX unique (CLIP + classifieur), 2 sorties : proba « IA » + embedding (pour le drift).
- Taille : **352 Mo** ; latence : **≈ 52 ms par image** sur CPU (110 ms mesurés quand le PC était chargé par un autre calcul).
- Vérification : écart ONNX vs PyTorch < 0,0001.
- MLflow : modèle `real-or-ai` version 1, alias `champion`.

## Niveau 0 — système complet en local (mesuré le 2026-11-08)

- 4 conteneurs lancés par `docker compose up` : MLflow, API (FastAPI + ONNX Runtime), Prometheus, Grafana.
- Démarrage de l'API : téléchargement du modèle (352 Mo) depuis MLflow + chargement ≈ **3 s**.
- Test de bout en bout : 400 images (50 par catégorie) envoyées une par une → 400 réponses en 32 s.
  - Latence côté serveur : **médiane 64 ms, p95 94 ms** par image (prétraitement + modèle).
  - Scores identiques à l'entraînement (aux variations près sur 50 images) : vraies photos 96 %, BigGAN 100 %, Midjourney 66 %, Wukong 54 % → le comportement est le même en production qu'à l'entraînement.

## Niveau 1 — sécurité, suivi d'usage, test de charge (mesuré le 2026-11-08)

### Sécurité par clé API
- Sans clé → **401** ; clé inconnue ou révoquée → **401** ; clé valide mais non admin sur `/admin/usage` → **403**.
- En base : seule l'empreinte SHA-256 de la clé est stockée (64 caractères hexadécimaux), jamais la clé.
- Cache des clés : 60 s → une seule requête SQL par clé et par minute au lieu d'une par appel.

### Test de charge Locust
Montée progressive de 0 à 40 utilisateurs virtuels en 160 s (1 nouvel utilisateur toutes les 4 s), chacun envoie une image puis attend 0,5 à 1,5 s. Durée 3 min. API = 1 seul conteneur, 1 processus. ⚠️ Locust tourne sur le même PC que l'API et lui prend du CPU.

| Utilisateurs | Requêtes/s | Latence médiane | Latence p95 |
|---|---|---|---|
| 4 | 1,9 | 71 ms | 120 ms |
| 12 | 8,9 | 100 ms | 190 ms |
| 19 | 14,3 | 180 ms | 550 ms |
| 23 | 15,9 | 290 ms | 980 ms |
| 30 | 18,8 | 560 ms | 1 100 ms |
| 40 | 20,0 | 1 100 ms | 1 600 ms |

- Total : **2 517 requêtes, 0 erreur (0,00 %)**.
- **Capacité maximale ≈ 20 requêtes/s** : au-delà de ~23 utilisateurs, le débit plafonne et ce sont les temps de réponse qui augmentent (les requêtes font la queue).
- Zone confortable : **jusqu'à ~12 utilisateurs simultanés (≈ 9 req/s) avec p95 < 200 ms**.
- `/usage/me` (lecture SQL seule) : médiane 36 ms → c'est bien le modèle qui coûte, pas la base.
- Point de départ pour le niveau 3 (quantization, scaling horizontal) : chiffres à battre.
