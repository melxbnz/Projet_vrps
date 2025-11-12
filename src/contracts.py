#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module des Contrats (Étape 1 et 2).
Définit les structures de données standards (dataclasses)
utilisées par tous les autres modules du projet (Instance, Solution).

NE DOIT PAS ÊTRE MODIFIÉ sans l'accord de l'équipe.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --- Types de base ---

# Un NodeId est un entier. 0 représente le dépôt.
NodeId = int

# Une Route est une liste d'ID de nœuds, commençant et finissant par 0.
# Exemple: [0, 3, 1, 0] (Dépôt -> Client 3 -> Client 1 -> Dépôt)
Route = List[NodeId]

# --- Définition d'une Instance ---

@dataclass
class Instance:
    """
    Structure de données immuable représentant une instance (un problème).
    Contient toutes les données lues depuis un fichier (ex: C101.txt).
    """
    name: str
    distance_matrix: List[List[float]]
    demand: List[int]                 # Demande pour chaque nœud (demand[0] == 0)
    capacity: int                     # Capacité max d'un véhicule
    
    # Champs optionnels (pour VRPTW)
    ready_time: Optional[List[float]] = None  # Heure d'ouverture (TW)
    due_time: Optional[List[float]] = None    # Heure de fermeture (TW)
    service_time: Optional[List[float]] = None # Temps passé chez le client

    # Champs optionnels (contraintes globales)
    Kmax: Optional[int] = None      # Nombre maximum de véhicules
    Tmax: Optional[float] = None    # Durée maximale d'une route

# --- Définition d'une Solution ---

@dataclass
class Solution:
    """
    Structure de données mutable représentant une solution.
    Elle est créée par 'initial_solution' et modifiée par 'optimization_loop'.
    Elle est lue par 'evaluation'.
    """
    routes: List[Route] = field(default_factory=list)
    cost: float = float("inf")
    feasible: bool = False
    
    # 'meta' est un dictionnaire pour stocker des infos annexes
    # (ex: temps de calcul, nombre d'itérations...)
    meta: Dict[str, float] = field(default_factory=dict)
    
    def copy(self):
        """
        Crée une copie profonde de la solution.
        Crucial pour l'ALNS et les optimisations.
        """
        import copy
        return copy.deepcopy(self)