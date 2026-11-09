# Texte à dire — « Vraie photo ou image IA ? »

Durée visée : 10 minutes environ (avec les 2 nouvelles slides). Phrases courtes, parler lentement, regarder le jury.
Si le temps est limité, raccourcir les slides 3 et 4.

---

## Slide 1 · Titre (30 s)

Bonjour, je m'appelle Naim Fertas.
Mon projet répond à une question simple : **cette image est-elle une vraie photo, ou a-t-elle été créée par une IA ?**
Mais le vrai sujet, ce n'est pas seulement le modèle. C'est tout ce qu'il faut pour le **mettre en service, le surveiller et le mettre à jour** quand il devient dépassé. C'est ça, le MLOps.

## Slide 2 · Le sujet (45 s)

Petit jeu pour commencer : parmi ces six images, trois n'ont jamais existé.
*(laisser 2 secondes)*
À l'œil, c'est difficile, surtout avec les IA récentes comme Midjourney.
Mon service fait ça automatiquement : on envoie une image, il répond **« vraie » ou « IA »**, avec un pourcentage de confiance.

## Slide 3 · Pourquoi ce sujet (45 s)

J'ai choisi ce sujet pour trois raisons.
Un : c'est **un vrai besoin** — fausses infos, arnaques, faux profils.
Deux, la plus importante : **les IA changent vite**. Un détecteur entraîné aujourd'hui sera dépassé demain. Il faut donc le surveiller et le mettre à jour : c'est exactement le rôle du MLOps.
Trois : c'est **facile à montrer**. Une image entre, une réponse sort.

## Slide 4 · La méthode (45 s)

Ma méthode tient en quatre étapes, qui tournent en boucle :
**Apprendre** : le modèle étudie des milliers d'exemples.
**Servir** : on le met en ligne.
**Surveiller** : une alarme sonne si les images changent.
**Mettre à jour** : un nouveau modèle, testé prudemment.
Puis on recommence. Je vais montrer chaque étape avec un chiffre.

## Slide 5 · Étape 1 — Apprendre (1 min)

Le modèle a appris sur **35 000 images** : moitié vraies photos, moitié images de 7 IA.
Un piège : toutes les vraies photos étaient en JPEG, les fausses en PNG. Le modèle aurait pu tricher en regardant le format. J'ai donc tout converti au même format.
Résultat : **92 % de bonnes réponses** sur des images jamais vues. Le hasard, c'est 50 %.
Et j'ai **caché exprès deux IA**, Midjourney et Wukong, pour simuler plus tard l'arrivée d'une IA inconnue.

## Nouvelle slide (après la 5) · Le modèle (45 s)

Comment le modèle fonctionne ? Il a deux morceaux.
D'abord **CLIP**, un modèle d'OpenAI qui a déjà vu **400 millions d'images**. Ce sont les « yeux » : il transforme chaque image en une empreinte de **512 nombres**. On ne le modifie pas.
Ensuite un **petit classifieur**, le « cerveau » : il lit l'empreinte et répond « vraie » ou « IA ». C'est la seule partie que j'ai entraînée, en quelques secondes.
Pourquoi ce choix ? On réutilise ce que CLIP sait déjà : pas besoin de millions d'images ni de carte graphique. Ça s'appelle le *transfer learning*.

## Slide 6 · Étape 2 — Servir (1 min)

Un modèle sur mon PC ne sert à personne. Il faut le mettre en ligne.
Tout le système se lance en **une seule commande**, chaque partie dans sa boîte — un conteneur — et il est installé automatiquement sur un serveur.
Chaque utilisateur a **une clé personnelle** : sans clé, accès refusé, et chaque requête est comptée.
Ici, j'envoie une vraie photo de camping-car : réponse **vraie photo à 99 %**.
J'ai aussi simulé **30 utilisateurs en même temps : zéro erreur**.

## Slide 7 · Étape 3 — Surveiller (1 min 15)

Problème : en production, **on ne connaît pas la bonne réponse**. On ne voit donc pas directement que le modèle se trompe.
Alors on surveille autre chose : est-ce que les images reçues ressemblent encore à celles de l'apprentissage ? Quand elles changent, on appelle ça le **data drift**.
Toutes les 30 secondes, le système calcule un score.
Images habituelles : **0,60**, tout est calme.
J'envoie les deux IA cachées : **0,81**. Le seuil est à 0,70 : **l'alarme sonne**.

## Nouvelle slide (après la 7) · Comment l'alarme marche (1 min)

Comment on calcule ce score ?
On garde **2 000 images de l'apprentissage** comme référence.
Toutes les 30 secondes, on prend les **200 dernières images reçues**.
Puis un petit « juge » essaie de deviner, pour chaque image : est-ce qu'elle vient de l'apprentissage, ou d'aujourd'hui ?
S'il devine au hasard, score **0,5** : les images n'ont pas changé.
S'il trouve facilement, score proche de **1** : elles ont changé. Au-dessus de **0,70**, l'alarme sonne.
L'avantage : **pas besoin de connaître la bonne réponse**, on regarde seulement les images.

## Slide 8 · Étape 4 — Mettre à jour (1 min)

J'ai réentraîné le modèle avec les 7 IA : c'est la version 2.
Mais on ne remplace pas un modèle d'un coup. On fait un **déploiement canary** : la v2 reçoit d'abord **10 % des requêtes**.
Le nom vient des mineurs, qui descendaient un canari dans la mine pour détecter le gaz avant que tout le monde soit touché.
Tout allait bien, donc je l'ai mise en service : le service la charge tout seul en **32 secondes, sans coupure**.

## Slide 9 · Le résultat principal (45 s)

Voici le chiffre qui résume tout le projet.
Au départ : **91 %** de bonnes réponses.
Les nouvelles IA arrivent : on tombe à **73 %**… mais l'alarme sonne, donc on le sait.
Après la mise à jour : **89 %**.
Le système **a vu le problème et s'est réparé**. Tous ces chiffres ont été mesurés sur le serveur.

## Slide 10 · Bonus — Plus rapide (45 s)

Pour aller plus loin, j'ai allégé le modèle en arrondissant ses nombres : ça s'appelle la quantization.
Il est **4 fois plus petit**, pour seulement **0,8 point** de précision en moins.
Avec deux copies du service, la capacité **double** : de 9 à 18 requêtes par seconde, sans erreur.

## Slide 11 · Conclusion (45 s)

Pour conclure : j'ai construit un détecteur qui **sait quand il vieillit, et se met à jour sans coupure**.
Les quatre niveaux du sujet sont couverts — au niveau 2, j'ai fait trois options sur quatre, il manque Terraform.
Ce que je peux améliorer : repérer des changements plus discrets, et répartir le service sur plusieurs machines.
Merci. Je réponds à vos questions.

---

## Slides 12 et 13 · « Image Sources »

Ne pas les présenter : les supprimer ou les mettre à la fin, après « Merci ».

---

## Si le jury pose une question

| Question probable | Réponse courte |
|---|---|
| Pourquoi ce modèle ? | CLIP connaît déjà très bien les images. Il suffit d'ajouter un petit classifieur : rapide à entraîner, et 92 % de bonnes réponses. |
| C'est quoi le data drift ? | Les données reçues ne ressemblent plus à celles de l'entraînement. Le modèle risque de se tromper. |
| Comment tu le détectes ? | Toutes les 30 s, un petit modèle essaie de distinguer les images récentes des images d'entraînement. S'il y arrive trop bien (score > 0,70), c'est que les images ont changé. |
| C'est quoi un canary ? | Donner la nouvelle version à 10 % des utilisateurs d'abord. Si elle a un souci, peu de gens sont touchés. |
| Et si la nouvelle version est mauvaise ? | On remet l'ancienne en un clic, sans coupure (retour en arrière). |
| Pourquoi pas de Terraform ? | La machine a été créée à la main. J'ai préféré faire trois autres options du niveau 2 en entier. |
| Les limites ? | Un changement discret (environ 25 % d'images nouvelles) n'est pas détecté. Et tout tourne sur une seule machine. |
| Comment tu as eu tes chiffres ? | Tout a été mesuré, en local et sur le serveur. |
