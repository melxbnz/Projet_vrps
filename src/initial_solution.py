"""
Étape 3 — Construction initiale (Romain)
Options: Clarke-Wright capacitaire / Insertion gloutonne time-aware.
"""
from .contracts import Instance, Solution

def build_initial_solution(instance: Instance, method: str = "clarke-wright") -> Solution:
    """
    Retourne une solution faisable (routes + coût approximatif).
    Respecter capacité, Kmax, fenêtres de temps.

    TODO: Romain — implémenter la heuristique choisie.
    """
    raise NotImplementedError("À implémenter par Romain")
