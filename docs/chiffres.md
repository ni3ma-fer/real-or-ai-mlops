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

## Niveau 1 — production sur VM (mesuré le 2026-11-08)

VM de production : Ubuntu 24.04 créée avec Multipass sur Hyper-V (sur le PC), **2 vCPU, 4 Go RAM, 25 Go disque**. La VM du prof (Azure) était injoignable.

### Déploiement en une commande (`scripts/deploy.sh ubuntu@<IP> --ship-images`)
- Premier déploiement sur VM vierge : **13 min 42 s** au total, dont l'installation de Docker par internet (connexion lente ~200 Ko/s) pour l'essentiel.
- Copie des 5 images Docker du PC vers la VM (~4 Go) : **1 min 49 s** (API 19 s, Prometheus 15 s, Grafana 25 s, MLflow 35 s, PostgreSQL 15 s) — au lieu de >1 h de téléchargement.
- Promotion du modèle dev → prod (`scripts/promote_model.sh`, tunnel SSH) : **29 s**, modèle chargé automatiquement par l'API.

### Sécurité vérifiée depuis l'extérieur
- API sur le port 80 : `/predict` sans clé → **401** ; avec clé → prédiction identique au dev (Midjourney : 0,1346).
- Ports MLflow 5000, Grafana 3000, Prometheus 9090, PostgreSQL 5432 : **fermés** depuis l'extérieur (accessibles seulement par tunnel SSH).
- Mots de passe de prod générés aléatoirement sur la VM (`openssl rand`), jamais dans Git.

### Test de charge sur la production
Locust lancé depuis le PC vers la VM, 0 → 30 utilisateurs en 150 s, 3 min.

| Utilisateurs | Requêtes/s | Latence médiane | Latence p95 |
|---|---|---|---|
| 4 | 2,0 | 150 ms | 190 ms |
| 10 | 7,4 | 160 ms | 330 ms |
| 13 | 8,5 | 300 ms | 1 000 ms |
| 18 | 9,7 | 770 ms | 1 400 ms |
| 30 | 9,7 | 2 400 ms | 3 500 ms |

- **1 430 requêtes, 0 erreur.**
- **Capacité de la prod ≈ 10 requêtes/s** (2 vCPU) contre ≈ 20 req/s en local (6 CPU pour Docker) → la capacité suit le nombre de cœurs : argument pour le scaling horizontal (niveau 3).
- Zone confortable : **~10 utilisateurs simultanés, p95 < 350 ms**.
- Suivi d'usage en prod : alice 682 requêtes, bob 629 → comptées par utilisateur dans PostgreSQL.

## Niveau 2 — interface, drift, MLflow (mesuré le 2026-11-08)

### Détection de data drift (méthode « domain classifier », comme Evidently pour les embeddings)
- Référence : 2 000 embeddings CLIP du jeu d'entraînement du modèle champion (artefact MLflow).
- Toutes les 30 s : les 200 dernières images reçues (dédoublonnées) vs la référence → AUC d'un classifieur logistique (validation croisée 5 plis). 0,5 = indiscernable, 1 = totalement différent.
- Calibration du seuil (fenêtres de 200 images sans doublons) :

| Trafic | Score |
|---|---|
| Normal (vraies photos + 5 IA connues) | 0,55 – 0,64 |
| 50 % Midjourney/Wukong | 0,72 – 0,76 |
| 70 % Midjourney/Wukong | 0,81 – 0,83 |
| 25 % inconnus seulement | ≈ 0,53 (non détecté : limite de la méthode) |

- **Seuil retenu : 0,70.** Piège rencontré : les images envoyées plusieurs fois gonflaient le score (0,69 au lieu de 0,60, fausse alerte) → dédoublonnage.
- Démo en conditions réelles (simulateur de trafic → API → base → job de drift) :
  - trafic normal : **score 0,606 → pas d'alerte** ; modèle v1 : **94 %** de bonnes réponses ;
  - trafic « nouvelles IA » (70 % Midjourney/Wukong) : **score 0,826 → DRIFT**, alerte Prometheus `DataDrift` ; modèle v1 : **74,5 %** (Wukong 52 %).

### Réaction : modèle v2, canary, promotion à chaud
- v2 = réentraînée avec les 7 générateurs : Midjourney 77,6 % → **92,0 %**, Wukong 57,8 % → **86,0 %**, vraies photos 92,7 % → 90,8 % (compromis).
- Canary : v2 avec l'alias `challenger`, **10 % du trafic** → mesuré **31 requêtes sur 300 (10,3 %)** vers la v2.
- Promotion : alias `champion` déplacé sur v2 dans MLflow → **l'API charge la v2 en 17 s, sans redémarrage** (vérification toutes les 30 s).
- Même trafic « nouvelles IA » : **74,5 % (v1) → 88,5 % (v2)** de bonnes réponses.
- Le score de drift reste élevé (0,78) avec la v2 : le trafic contient 70 % de Midjourney/Wukong contre 14 % dans l'entraînement de la v2. Le drift mesure « les données ont changé », la performance dit si le modèle s'en sort.

### Interface utilisateur
- Streamlit (port 8501) : analyse d'une image, consommation personnelle, onglet surveillance (drift) réservé aux admins. L'interface n'a pas de modèle : elle appelle l'API avec la clé de l'utilisateur.

## Niveau 3 — quantization, scaling horizontal, canary (mesuré le 2026-11-08)

### Quantization INT8 (ONNX Runtime, dynamique, couches MatMul)
| | float32 (v2) | INT8 (v4) |
|---|---|---|
| Taille | 352 Mo | **96 Mo (3,7x plus petit)** |
| Latence par image (CPU) | 60 ms | **37 ms (1,6x plus rapide)** |
| Bonnes réponses (800 images de validation) | 92,9 % | **92,1 % (−0,8 pt)** |
| Accord des prédictions fp32/int8 | | 94 % |

- Problème rencontré : sans correction, l'INT8 décalait les scores vers « IA » → vraies photos **92 % → 63 %**. La quantization « par canal » était pire (0 %).
- Solution : **recalibration** (Platt) sur 1 000 images d'**entraînement** : 2 paramètres ajoutés avant la sigmoïde finale (logit' = 1,734 × logit − 2,009) → vraies photos 90 %.

### Tests de charge comparés (local, montée à 40 utilisateurs, 2 min)
| Configuration | Débit au plateau | Latence médiane | p95 | Erreurs |
|---|---|---|---|---|
| v2 float32, 1 copie | 17,3 req/s | 810 ms | 2 400 ms | 0 |
| **v4 INT8, 1 copie** | **23,7 req/s (+37 %)** | **250 ms** | 1 200 ms | 0 |
| v4 INT8, 3 copies (2 cœurs chacune) | 22,9 req/s | 360 ms | 1 300 ms | 0 |

- Répartition nginx entre les 3 copies : 629 / 629 / 605 requêtes (round robin).
- **3 copies sur une seule machine = pas plus de débit** : elles partagent les mêmes CPU. Le scaling horizontal augmente la capacité quand les copies sont sur des machines différentes (ex. nœuds Kubernetes).
- **Résilience** : une copie tuée en plein test de charge → 1 erreur sur 750 (0,13 %) ; avec le rejeu automatique nginx (`proxy_next_upstream`) → **0 erreur sur 701**.

## Démo complète rejouée en PRODUCTION (VM, 2026-11-08 soir)

| Étape | Résultat |
|---|---|
| 1. Trafic normal (v1) | drift **0,597 → ok**, 91,5 % de bonnes réponses |
| 2. Arrivée de nouvelles IA (v1) | drift **0,811 → DRIFT**, 73,5 % (Midjourney 69 %, Wukong 59 %) |
| 3. Canary : v2-INT8 en `challenger`, 10 % du trafic | **32 / 300 requêtes (10,7 %)** vers le challenger |
| 4. Promotion `challenger → champion` (scripts/set_alias.sh) | API passée sur le nouveau modèle **en 32 s, sans redémarrage** |
| 5. Même trafic « nouvelles IA » | **89,0 %** de bonnes réponses (Midjourney 95 %, Wukong 85 %), 200 images en 13 s au lieu de 17 s |

### Capacité de la production (VM 2 vCPU, Locust depuis le PC, 0 → 30 utilisateurs)
| | v1 float32, 1 copie | v2 INT8, 2 copies (1 cœur chacune) |
|---|---|---|
| 10 utilisateurs | 7,5 req/s, médiane 190 ms | 7,8 req/s, **médiane 92 ms** |
| 20 utilisateurs | 9,4 req/s, médiane 1 000 ms | 13,2 req/s, médiane 150 ms |
| 30 utilisateurs | 9,2 req/s, médiane 2 250 ms | **18,1 req/s, médiane 430 ms** |
| Erreurs | 0 / 1 430 | 0 / 1 745 |

→ **Capacité de la prod ×2** (≈ 9 → ≈ 18 req/s), latence médiane ÷5 à forte charge.

### Incident de production résolu
- Après ajout des services, la VM (4 Go) saturait : charge système 36, 9 Mo de RAM libre, timeouts.
- Cause trouvée avec `docker stats` : **MLflow 3 = 2 Go** (4 workers + ~10 processus de « jobs » GenAI).
- Correctif : `--workers 1` + `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` → **MLflow 2 Go → 419 Mo**, charge système revenue à ~2.
