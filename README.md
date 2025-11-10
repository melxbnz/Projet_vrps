# Projet VRP/VRPTW — CESI

Ce dépôt contient la structure du projet pour l'étude et l'implémentation d'une métaheuristique (ALNS/VND) sur des instances VRP/VRPTW.

## Structure
- `src/` — modules par étape (une personne par fichier)
- `notebook/projet_final.ipynb` — démonstration & visualisation
- `data/` — instances VRPLIB (non versionnées)
- `results/` — logs & figures générés automatiquement
- `tests/` — tests rapides (smoke tests)
- `scripts/` — scripts d'exécution

## Environnement
Créer l'environnement conda :
```bash
conda env create -f environment.yml
conda activate projet-vrp
```

## Lancement rapide
```bash
python scripts/run_main.py
```

## Conventions d'équipe
- Une branche par personne : `melissa/evaluation`, `carelle/neighborhoods`, `olivier/alns`, `romain/initial_solution`.
- Interface stable (types I/O) entre modules (décrite dans `src/contracts.py`).

