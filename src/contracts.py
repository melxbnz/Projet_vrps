"""
Contrats d'interface — types attendus par chaque module.

Ne pas implémenter d'algorithme ici. On définit seulement les structures
minimales pour que chacun colle son code en gardant la compatibilité.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

NodeId = int  # 0 = dépôt
Route = List[NodeId]  # ex: [0, 5, 3, 0]

@dataclass
class Instance:
    name: str
    distance_matrix: List[List[float]]
    demand: List[int]                # demande par nœud (0 pour dépôt)
    capacity: int                    # capacité camion (1D pour base)
    ready_time: Optional[List[float]] = None  # TW: ouverture
    due_time: Optional[List[float]] = None    # TW: fermeture
    service_time: Optional[List[float]] = None
    Kmax: Optional[int] = None       # nb de camions max
    Tmax: Optional[float] = None     # retour max dépôt (temps)    

@dataclass
class Solution:
    routes: List[Route] = field(default_factory=list)
    cost: float = float("inf")
    feasible: bool = False
    meta: Dict[str, float] = field(default_factory=dict)  # infos diverses (temps, itérations...)
