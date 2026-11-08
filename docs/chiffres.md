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
