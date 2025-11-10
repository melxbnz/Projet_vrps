"""
Étape 7 — Boucle d'optimisation (Melissa)
Boucle principale pilotant l'ALNS/VND + trace de convergence.
"""
from typing import Dict, List
from .contracts import Instance, Solution
from .alns import alns_step

def optimization_loop(instance: Instance, init_solution: Solution, max_iter: int = 800) -> Dict[str, List[float]]:
    """
    Returns
    -------
    history : dict
        {"iter": [...], "cost_current": [...], "cost_best": [...]}
    """
    # TODO: Melissa — boucle, MAJ solution courante / best, remplir history
    raise NotImplementedError
