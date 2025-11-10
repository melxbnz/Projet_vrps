"""
Étape 1 — Chargement des données (Alban)
But : transformer une instance VRPLIB en `Instance` utilisable par tous.
"""
from typing import Any
from .contracts import Instance

def load_instance(path_or_name: str) -> Instance:
    """
    Parameters
    ----------
    path_or_name : str
        Chemin local vers .vrp ou nom VRPLIB (ex: "A-n32-k5.vrp")

    Returns
    -------
    Instance
        Objet standardisé contenant matrice des distances, demandes, etc.
    """
    # TODO: Alban — implémenter via vrplib (fallback: lecture locale)
    raise NotImplementedError("À implémenter par Alban")
