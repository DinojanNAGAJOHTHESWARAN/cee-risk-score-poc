# Architecture technique

Le POC combine trois composantes complémentaires :

1. **Modèle supervisé** — régression logistique pondérée, entraînée uniquement sur les dossiers disposant d'une cible qualifiée. Elle estime l'association avec un résultat de contrôle historiquement défavorable.
2. **Détection d'anomalies** — Isolation Forest, utilisée pour repérer des dossiers atypiques sans transformer l'atypisme en preuve de fraude.
3. **Règles métier** — signaux explicites séparés en facteurs de risque, anomalies et incertitudes. Une information manquante ou un historique insuffisant n'ajoute pas automatiquement de risque.

Le score hybride du POC est `0.40 × supervisé + 0.35 × métier + 0.25 × anomalie`. Les poids et seuils 40/70 sont des hypothèses de prototype à calibrer avec les équipes métier.

## Prévention des fuites de données

Les variables créées après contrôle (résultat, statut final, SAV, motifs de rejet, etc.) servent éventuellement à construire la cible d'entraînement mais sont exclues des prédicteurs. Le flux d'import détecte et retire également ces colonnes avant scoring.

## Validation

La séparation train/test est temporelle afin de se rapprocher d'un usage futur. Les métriques suivies sont ROC-AUC, PR-AUC, recall, précision et Recall/Precision@Top-k, ces dernières étant particulièrement adaptées à une capacité de contrôle limitée.

## Human-in-the-loop

Le score ordonne une file de contrôle. Il ne déclenche aucun rejet automatique et ne constitue pas une qualification juridique de fraude.
