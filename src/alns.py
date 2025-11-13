# Olivier
# =============================================================================
# IMPLEMENTATION DE L'ALNS POUR LE VRPTW-B - VERSION MODULAIRE ADAPTÉE
# =============================================================================
# Auteur: Grok (expert en optimisation combinatoire et métaheuristiques)
# Date: 13/11/2025
# Description:
# Adaptation modulaire d'alns.py pour intégrer pleinement neighborhoods.py (apply_two_opt/relocate/swap,
# delta_relocate/swap, delta_cost wrapper pour relocate/swap). Évite redondance: Remplacement des fonctions
# custom two_opt_move/relocate_move/swap_move par wrappers utilisant apply_* et delta_* (O(1) via evaluation
# pour two_opt). generate_candidates utilise ces wrappers (modulaire: imports relatifs .neighborhoods).
# Alignement signatures: Instance/Solution/Route de contracts.py ; evaluate/check/compute/delta_two_opt de evaluation.py.
# Uniformité Notebook/PDF: Variables q_i=demand[i], e_i=ready_time[i], l_i=due_time[i], s_i=service_time[i],
# Q_k=capacity, Kmax, Tmax ; F=∑ c_ij x_ijk min sous contraintes (Partie 2-4). Moves: indices internes
# (1<=i<j<len(r)-1 pour two_opt ; excl dépôts 0). Feasible post-move via check_feasibility (TW attente,
# Q sum[1:-1], Tmax current<=Tmax). Delta: O(1) formules (rm/ins pour relocate, new-old pour swap,
# c_{i-1 j}+c_{i j+1}-c_{i-1 i}-c_{j j+1} pour two_opt).
# Flux PDF (Parties 5-7): Voisinages (5: gen k=10 cands random/valides, tri delta asc) → ALNS (6: destroy
# fraction=0.3 clients random, repair greedy tri e_i min delta approx, accept hiérarchique/metro T=100,
# VND descent ordre TWO→RELOC→SWAP while imp delta<0/aspir feas<best, shake 2-5 reloc si stagn>50,
# adapt EMA α=0.1 window=10 *10 scale) → Boucle (7: 2000 iters, trace every 100, penalty dyn *0.99/1.1).
# Changements: __init__ eval initial ; destroy/repair score=1 si feas (destroy) / feas&cost<curr (repair) ;
# accept: feas&<curr yes, +pen<curr yes, else exp(-Δ/100) ; vnd: k=5 cands, aspiration global ;
# run_iter: True si best imp. Modularité: Imports .contracts/.evaluation/.neighborhoods ; pas redondance
# (deltas inline neighborhoods, feasible eval). Performances: O(2000 * n^2) scalable (deltas O(1),
# checks O(len r) ~O(n/|routes|)). Erreurs: ValueError indices apply (via neighborhoods), inf delta
# (!feas/Q viol), logs unassigned/shake/penalty. Test: C101_small adapté (n=5, truncate load).
# Structure: Alignée PDF/Notebook ; commentaires exhaustifs (techniques/lignes: pré/post/complexité/uniformité).
# =============================================================================

# =============================================================================
# 1. IMPORTATIONS ET CONSTANTES (Modulaires: .neighborhoods pour apply/delta ; Partie 5-6)
# =============================================================================
import numpy as np  # Moyennes adapt_weights (EMA hist)
import random  # Aléa moves/destroy/repair (uniform weights, sample pos)
from dataclasses import dataclass  # Erreur orig: from dataclass → dataclasses ; inline si besoin
from copy import deepcopy  # Copies Solution (current/best ; .copy() contracts)
from enum import Enum  # MoveType (two_opt/relocate/swap ; Partie 5)
from typing import List, Tuple, Optional, Dict  # Annotations (List[Route], etc. ; aligné Notebook)

# Imports modulaires (relatifs ; cf. PDF Partie 2/4/5 ; uniformité vars)
from .contracts import Instance, Solution, Route  # Structures (Instance: q_i=demand, e_i=ready_time, etc.)
from .evaluation import evaluate_solution, check_feasibility, compute_route_cost, delta_cost_two_opt  # Éval (F, feasible TW/Q/Tmax, delta two_opt O(1))
from .neighborhoods import (  # Voisinages (Partie 5: apply in-place, delta O(1) ; modulaire)
    apply_two_opt, apply_relocate, apply_swap,  # Appliquer moves (modif routes)
    delta_relocate, delta_swap, delta_cost  # Deltas (wrapper relocate/swap ; two_opt via eval)
)

# Constantes (Partie 6-7 PDF ; ajusté scalabilité Notebook: max_iter=2000 >800 pour conv)
PENALTY_FACTOR = 1000.0  # Pénalité init infaisables (dynamique: *0.99 improve feas, *1.1 stagn/10)
SEGMENT_SIZE = 0.3  # Fraction destroy/repair (Partie 6: ~30% |Vc| clients modifiés)
WEIGHT_UPDATE = 0.1  # Alpha EMA adaptation (Partie 6: (1-α)old + α mean(hist[-10])*10)
NO_IMPROVEMENT_LIMIT = 50  # Seuil shake (Partie 6.3: diversification si no_improv >=50)

class MoveType(Enum):
    """Enum types moves (Partie 5 PDF/Notebook: aligné str values pour delta_cost wrapper)."""
    TWO_OPT = "two_opt"  # Inversion segment [i:j+1] intra-route (delta O(1) via evaluation)
    RELOCATE = "relocate"  # Déplacement client route k1 pos i → k2 pos j (inter ; delta rm+ins)
    SWAP = "swap"  # Échange clients k1 i ↔ k2 j (inter ; delta new-old, simplifié voisins |i-j|=1)

# =============================================================================
# 5. VOISINAGES ADAPTÉS - WRAPPERS POUR MODULARITÉ (Utilise .neighborhoods apply/delta ; Partie 5)
# =============================================================================
def two_opt_move(route: Route, i: int, j: int, instance: Instance) -> Tuple[Route, float]:
    """
    Wrapper two-opt modulaire (Partie 5.1: utilise delta_cost_two_opt evaluation + apply_two_opt neighborhoods).
    Pré: 1 <= i < j < len(route)-1 (internes ; pas dépôts via validation apply).
    Post: new_route (inversion via apply on temp), delta O(1) ; inf si indices KO ou !feasible post (TW/Q/Tmax).
    Flux: Delta → apply temp sol → check_feasibility → return (copie si ok).
    Complexité: O(1) delta + O(len(route)) apply/check (scalable).
    Uniformité Notebook: Indices absolus [0,c1,...,0] ; feasible inclut attente max(e_v, current+travel)<=l_v +s_v.
    Erreurs: ValueError indices (propagé apply) ; inf !feas (retard/Q viol).
    """
    # Delta O(1) via evaluation (formule c_{i-1 j} + c_{i j+1} - c_{i-1 i} - c_{j j+1})
    try:
        delta = delta_cost_two_opt(route, i, j, instance.distance_matrix)
    except ValueError:
        return route[:], float('inf')  # Indices KO
    if delta == float('inf'):
        return route[:], float('inf')
    # Apply modulaire (temp sol un route ; in-place ok car temp)
    new_route = route[:]  # Copie
    temp_sol = Solution(routes=[new_route])  # Temp pour apply
    try:
        apply_two_opt(temp_sol, 0, i, j)  # Applique inversion [i:j+1]
    except ValueError:
        return route[:], float('inf')  # Validation échouée
    new_route = temp_sol.routes[0]  # Récup post-apply
    # Vérif feasible post-move (Partie 4: TW cumul current_time, Q sum[1:-1], Tmax)
    if not check_feasibility(new_route, instance):
        return new_route, float('inf')  # Pénalise (accept gère via penalty)
    return new_route, delta  # Valide (delta<0 améliore souvent)

def relocate_move(solution: Solution, k1: int, i: int, k2: int, j: int, instance: Instance) -> Tuple[Solution, float]:
    """
    Wrapper relocate modulaire (Partie 5.2: utilise delta_relocate + apply_relocate neighborhoods).
    Pré: k1 != k2 typique, 1<=i<len(r1)-1, 0<=j<=len(r2) (fin ok via validation apply).
    Post: new_sol (copy + apply), delta O(1) rm+ins ; inf si indices/Q KO ou !feas affected routes.
    Flux: Delta (check Q inter via delta_relocate) → copy sol → apply → check k1/k2 → return.
    Complexité: O(1) delta + O(len(r2)) apply/insert + O(len(r1/r2)) check (scalable).
    Uniformité Notebook: u=r1[i] déplacé ; delta approx -c_a1b1 +c_a1u +c_ub1 (rm) + sym ins ; feasible post TW/Q.
    Erreurs: ValueError indices (propagé apply) ; inf !Q (sum r2[1:-1] + q_u > Q_k) / !feas.
    Note: Si r1 post <3, inf (évite vide ; adaptable pop route).
    """
    # Delta O(1) via neighborhoods (incl check Q inter: sum q r2 clients + q_u <= Q_k)
    try:
        delta = delta_relocate(solution, instance, k1, i, k2, j)
    except (ValueError, IndexError):
        return solution.copy(), float('inf')
    if delta == float('inf'):
        return solution.copy(), float('inf')  # Indices/Q KO
    # Apply modulaire (copy sol ; in-place apply)
    new_sol = solution.copy()
    try:
        apply_relocate(new_sol, k1, i, k2, j)  # Pop i k1 → insert j k2
    except ValueError:
        return solution.copy(), float('inf')  # Validation échouée
    # Vérif feasible affected routes (Partie 4: TW/Q/Tmax ; global post non requis ici)
    if len(new_sol.routes[k1]) < 3:  # Vide post-pop? Pénalise
        return new_sol, float('inf')
    if (not check_feasibility(new_sol.routes[k1], instance) or
        not check_feasibility(new_sol.routes[k2], instance)):
        return new_sol, float('inf')
    # Kmax si nouvelle? Ici inter existantes ; adaptable
    if len(new_sol.routes) > (instance.Kmax or len(instance.demand)-1):
        return new_sol, float('inf')
    return new_sol, delta  # Valide

def swap_move(solution: Solution, k1: int, i: int, k2: int, j: int, instance: Instance) -> Tuple[Solution, float]:
    """
    Wrapper swap modulaire (Partie 5.3: utilise delta_swap + apply_swap neighborhoods).
    Pré: k1 != k2 typique, 1<=i<len(r1)-1, 1<=j<len(r2)-1 (pas dépôts via validation).
    Post: new_sol (copy + apply), delta O(1) new-old ; inf si indices/Q KO ou !feas affected.
    Flux: Delta (check Q inter: r1 -q_u +q_v <=Q_k sym) → copy → apply échange → check k1/k2.
    Complexité: O(1) delta/apply + O(len(r1/r2)) check.
    Uniformité Notebook: u=r1[i] ↔ v=r2[j] ; delta simplifié si k1=k2 |i-j|=1 (3 arcs) else 4 arcs ;
    feasible post (ordre impact TW).
    Erreurs: ValueError indices (propagé) ; inf inutile (i==j même) / !Q / !feas.
    """
    # Delta O(1) via neighborhoods (incl check Q inter, cas voisins)
    try:
        delta = delta_swap(solution, instance, k1, i, k2, j)
    except (ValueError, IndexError):
        return solution.copy(), float('inf')
    if delta == float('inf'):
        return solution.copy(), float('inf')  # Indices/Q/inutile KO
    # Apply modulaire
    new_sol = solution.copy()
    try:
        apply_swap(new_sol, k1, i, k2, j)  # Échange direct
    except ValueError:
        return solution.copy(), float('inf')
    # Vérif affected
    if (not check_feasibility(new_sol.routes[k1], instance) or
        not check_feasibility(new_sol.routes[k2], instance)):
        return new_sol, float('inf')
    return new_sol, delta

# =============================================================================
# 6. GÉNÉRATION CANDIDATS ADAPTÉE - MODULAIRE (Utilise wrappers move ; Partie 5/6)
# =============================================================================
def generate_candidates(solution: Solution, instance: Instance, move_type: MoveType, k: int = 10) -> List[Tuple[Solution, float]]:
    """
    Génère k candidats modulaires (Partie 5/6: random pos/routes valides ; utilise wrappers *_move).
    Tri delta asc (meilleurs first pour VND). Oversample k*2 pour filtre inf/!feas.
    Pour TWO_OPT: intra-route i<j random (si len(r)>=4) ; RELOCATE/SWAP: inter k1!=k2, pos random.
    Post: evaluate_solution non (delta approx ; full en VND accept). Feasible via wrappers.
    Complexité: O(k * |routes|^2) worst (random sample ; scalable k=10).
    Uniformité Notebook: MoveType Enum → str pour compat ; candidats [(new_sol copy, delta)] top-k.
    Erreurs: [] si none valide (small n, !feas) ; logs aucun.
    Note: k=10 >5 orig pour diversité ; adaptable VND.
    """
    candidates = []
    routes = solution.routes
    mt_str = move_type.value  # Str pour compat delta_cost si besoin (mais wrappers gèrent)
    num_attempts = k * 2  # Oversample pour ~k valides

    if move_type == MoveType.TWO_OPT:
        # Intra-route: sample i<j si len>=4
        for _ in range(num_attempts):
            if not routes or all(len(r) < 4 for r in routes):
                break
            r_idx = random.randint(0, len(routes) - 1)
            route = routes[r_idx]
            seq_len = len(route)
            if seq_len < 4: continue
            i = random.randint(1, seq_len - 3)
            j = random.randint(i + 1, seq_len - 2)
            new_route, delta = two_opt_move(route, i, j, instance)
            if delta < float('inf'):
                new_sol = solution.copy()
                new_sol.routes[r_idx] = new_route
                candidates.append((new_sol, delta))

    elif move_type == MoveType.RELOCATE:
        # Inter: k1!=k2, i interne from, j 0..len to
        for _ in range(num_attempts):
            if len(routes) < 2: break
            k1 = random.randint(0, len(routes) - 1)
            if len(routes[k1]) < 3: continue
            i = random.randint(1, len(routes[k1]) - 2)
            k2 = random.choice([idx for idx in range(len(routes)) if idx != k1])
            j = random.randint(0, len(routes[k2]))  # Incl fin
            new_sol, delta = relocate_move(solution, k1, i, k2, j, instance)
            if delta < float('inf'):
                candidates.append((new_sol, delta))

    elif move_type == MoveType.SWAP:
        # Inter: k1!=k2, i/j internes
        for _ in range(num_attempts):
            if len(routes) < 2: break
            k1 = random.randint(0, len(routes) - 1)
            if len(routes[k1]) < 3: continue
            i = random.randint(1, len(routes[k1]) - 2)
            k2 = random.choice([idx for idx in range(len(routes)) if idx != k1])
            if len(routes[k2]) < 3: continue
            j = random.randint(1, len(routes[k2]) - 2)
            new_sol, delta = swap_move(solution, k1, i, k2, j, instance)
            if delta < float('inf'):
                candidates.append((new_sol, delta))

    # Tri asc delta (meilleurs moves first ; top-k)
    candidates.sort(key=lambda x: x[1])
    return candidates[:k]

# =============================================================================
# 7. CLASSE ALNS ADAPTÉE - INTÉGRATION MODULAIRE (Utilise generate_candidates ; Partie 6)
# =============================================================================
class ALNS:
    """
    ALNS adaptative modulaire (Partie 6 PDF/Notebook: large neighborhood search).
    Attributs: instance (données), current/best_solution (mutable via copy), destroy/repair_operators
    (Dict[MoveType,float] poids init=1.0), weights_history (Dict[MoveType,List[float]] scores 0/1),
    penalty (float dyn init=1000), no_improvement (int stagn).
    Méthodes: destroy (random fraction clients pop, score feas), repair (greedy insert tri e_i min delta approx,
    nouvelle route si <Kmax & feas), accept_solution (hiérarchique: feas&<curr yes, +pen<curr yes, metro exp(-Δ/100)),
    vnd_local_search (descent: ordre TWO→RELOC→SWAP, while imp delta<0/aspir feas<best.cost, k=5 cands),
    shake_solution (2-5 reloc random si >=50 stagn, reset=0), adapt_weights (EMA α=0.1 window=10 *10 scale, trim),
    run_iteration (flux destroy→repair→accept→VND→update best/penalty/no_improv→shake?→adapt ; True si best imp).
    Flux: Modularité via generate_candidates (voisins) ; evaluate/check (éval) ; feasible prefer (penalty guide).
    Complexité init: O(1) + O(n) eval initial ; iter: O(n^2) (destroy/repair O(n^2), gen k=10 O(|routes|^2)).
    Uniformité Notebook: Scores destroy=1 si post feas (rare), repair=1 si feas&cost<curr ; T=100 fixe metro ;
    aspiration VND: global best. Erreurs: Logs non trouvé client / non réinséré (TW/Q/Kmax) / shake.
    Note: Operators sym destroy/repair (même hist ; adaptable distincts).
    """
    def __init__(self, instance: Instance, initial_solution: Solution):
        """
        Initialisation ALNS (Partie 6: setup operators/poids/hist, copie current/best, eval pour cohérence).
        Pré: Instance (q_i,e_i,...), initial_solution (feasible? via CW Partie 3).
        Post: Attribs prêts ; current/best.cost/feasible/meta["nb_routes"] updatés.
        Complexité: O(taille solution) copy + O(n) eval (sum len routes).
        Uniformité: Poids=1.0 uniformes ; penalty=1000 guide infaisables vers feas.
        """
        self.instance = instance  # Données (c_ij, q_i, Q_k, e_i/l_i/s_i, Kmax, Tmax)
        self.destroy_operators = {mt: 1.0 for mt in MoveType}  # Poids destroy (adaptés EMA)
        self.repair_operators = {mt: 1.0 for mt in MoveType}  # Poids repair (sym ; adaptable)
        self.current_solution = initial_solution.copy()  # Courante (mutable moves)
        self.best_solution = initial_solution.copy()  # Meilleure (feas prefer)
        self.weights_history = {mt: [] for mt in MoveType}  # Hist scores (0/1 improve ; window=10)
        self.penalty = PENALTY_FACTOR  # Pénalité !feas (dyn: *0.99 imp feas, *1.1 stagn/10)
        self.no_improvement = 0  # Compteur stagn (shake si >=50)
        # Éval initiale pour cohérence (cost F, feasible global, meta nb_routes)
        evaluate_solution(self.current_solution, instance)
        evaluate_solution(self.best_solution, instance)

    def destroy(self, fraction: float = SEGMENT_SIZE) -> Solution:
        """
        Destruction partielle (Partie 6.1: sélection op roulette, remove ~fraction*|Vc| clients random).
        Pop pos via index ; cleanup [0,0] vide ; evaluate_solution post (update cost/feas/meta).
        Score history: 1 si post feasible (rare, approx preserve) else 0.
        Pré: current_solution (routes non vides).
        Post: destroyed (partiel unassigned ; |routes| réduit si vide).
        Complexité: O(fraction * n) sample + O(n) pops/eval (scalable).
        Uniformité Notebook: Clients 1..n random.sample ; logs non trouvé (rare).
        Note: Op sélection pondérée mais unique random ici (adapt hist pour futurs).
        """
        # Sélection opérateur pondérée (roulette ; simplifié random si weights uniformes)
        total_weight = sum(self.destroy_operators.values())
        rand_val = random.uniform(0, total_weight)
        cum_weight = 0.0
        selected_op = None
        for op, weight in self.destroy_operators.items():
            cum_weight += weight
            if rand_val <= cum_weight:
                selected_op = op
                break
        if selected_op is None:
            selected_op = random.choice(list(MoveType))  # Fallback
        # Copie + destruction
        destroyed = self.current_solution.copy()
        num_to_destroy = int(fraction * (len(self.instance.demand) - 1))  # ~30% |Vc|
        all_clients = list(range(1, len(self.instance.demand)))  # Vc=1..n
        to_remove = set(random.sample(all_clients, min(num_to_destroy, len(all_clients))))
        for client in to_remove:
            removed = False
            for r_idx in range(len(destroyed.routes) - 1, -1, -1):  # Reverse pour pop safe
                route = destroyed.routes[r_idx]
                if client in route[1:-1]:
                    pos = route.index(client)
                    destroyed.routes[r_idx] = route[:pos] + route[pos + 1:]
                    removed = True
                    # Cleanup vide [0,0]
                    if len(destroyed.routes[r_idx]) == 2 and destroyed.routes[r_idx] == [0, 0]:
                        destroyed.routes.pop(r_idx)
                    break
            if not removed:
                print(f"⚠️ Client {client} non trouvé dans routes (unassigned prior?).")  # Rare
        # Éval post-destroy (cost partiel, feasible? approx non)
        evaluate_solution(destroyed, self.instance)
        # Score pour adaptation (1 si feasible post, guide ops préserv feas)
        score = 1.0 if destroyed.feasible else 0.0
        self.weights_history[selected_op].append(score)
        return destroyed

    def repair(self, destroyed: Solution) -> Solution:
        """
        Reconstruction (Partie 6.1: sélection op roulette, insert unassigned tri e_i vers best pos min delta approx).
        Delta: insertion -old arc (O(1)) si check_feasible(temp) ; nouvelle route si <Kmax & feas.
        evaluate_solution post ; score=1 si feasible & cost < current (améliore).
        Pré: destroyed (unassigned set Vc - couverts).
        Post: repaired (réinséré max ; logs non-insérés TW/Q/Kmax).
        Complexité: O(|unassigned| * |routes| * avg len(r)) ~O(n^2) (scan pos/check O(len)).
        Uniformité Notebook: Tri ready_time e_i (time-aware) ; delta approx c_prev_client + c_client_next - c_prev_next.
        Note: Nouvelle route prior last resort ; op pondérée mais random ici.
        """
        # Sélection opérateur (roulette ; fallback random)
        total_weight = sum(self.repair_operators.values())
        rand_val = random.uniform(0, total_weight)
        cum_weight = 0.0
        selected_op = None
        for op, weight in self.repair_operators.items():
            cum_weight += weight
            if rand_val <= cum_weight:
                selected_op = op
                break
        if selected_op is None:
            selected_op = random.choice(list(MoveType))
        # Unassigned: set diff Vc - union r[1:-1] ; tri e_i ascending (time-aware)
        unassigned = set(range(1, len(self.instance.demand)))
        for route in destroyed.routes:
            unassigned -= set(route[1:-1])
        unassigned = sorted(unassigned, key=lambda client_id: self.instance.ready_time[client_id]
                            if self.instance.ready_time else 0.0)
        repaired = destroyed.copy()  # Base partielle
        for client in unassigned:  # Par ordre temps
            best_delta = float('inf')
            best_route_idx, best_pos = -1, -1
            # Scan routes existantes pour best insert
            for r_idx, route in enumerate(repaired.routes):
                for pos in range(1, len(route)):  # Pos après 0, avant fin (excl dernier 0)
                    temp_route = route[:pos] + [client] + route[pos:]
                    if check_feasibility(temp_route, self.instance):  # Feas temp (TW/Q/Tmax)
                        # Delta approx O(1): insertion cost - old arc
                        prev_node = route[pos - 1]
                        next_node = route[pos] if pos < len(route) - 1 else 0
                        old_cost = self.instance.distance_matrix[prev_node][next_node]
                        new_cost = (self.instance.distance_matrix[prev_node][client] +
                                    self.instance.distance_matrix[client][next_node])
                        approx_delta = new_cost - old_cost
                        if approx_delta < best_delta:
                            best_delta = approx_delta
                            best_route_idx, best_pos = r_idx, pos
            if best_route_idx != -1:  # Insert best existante
                repaired.routes[best_route_idx] = (repaired.routes[best_route_idx][:best_pos] +
                                                   [client] + repaired.routes[best_route_idx][best_pos:])
            else:  # Nouvelle route si possible
                max_routes = self.instance.Kmax or (len(self.instance.demand) - 1)
                if len(repaired.routes) < max_routes:
                    new_route = [0, client, 0]
                    if check_feasibility(new_route, self.instance):
                        repaired.routes.append(new_route)
                    else:
                        print(f"⚠️ Client {client} non réinséré: nouvelle route infaisable (TW/Q/Tmax viol).")
                else:
                    print(f"⚠️ Client {client} non réinséré: Kmax={max_routes} atteint.")
        # Éval globale post-repair (cost F, feasible, meta nb_routes)
        evaluate_solution(repaired, self.instance)
        # Score adaptation: 1 si améliore feasible & cost (guide repair efficaces)
        score = 1.0 if (repaired.feasible and repaired.cost < self.current_solution.cost) else 0.0
        self.weights_history[selected_op].append(score)
        return repaired

    def accept_solution(self, candidate: Solution) -> bool:
        """
        Critère acceptation hiérarchique (Partie 6: favor feasible/local opt).
        1. Feasible & cost < current: accept (améliore best pot).
        2. !Feasible mais cost + penalty < current: accept (guide vers feasible).
        3. Sinon: Metropolis exp(-Δ/T) diversification (T=100 fixe, light SA).
        Pré: Candidate évalué (cost/feasible via repair).
        Post: True accept (update current post-VND).
        Complexité: O(1).
        Uniformité Notebook: Δ = (cand.cost + pen if !feas) - current.cost ; prob=exp(-Δ/100) si >0.
        Note: Penalty dyn en run_iter (réduit imp feas, augmente stagn).
        """
        if candidate.feasible and candidate.cost < self.current_solution.cost:
            return True  # Améliore feasible
        cand_cost = candidate.cost + (self.penalty if not candidate.feasible else 0.0)
        if cand_cost < self.current_solution.cost:
            return True  # Améliore même infaisable (pénale violations)
        delta = cand_cost - self.current_solution.cost  # >0 dégradation
        prob_accept = np.exp(-delta / 100.0)  # T=100 (param fixe ; adaptable)
        return random.random() < prob_accept  # Alea diversification

    def vnd_local_search(self, init_candidate: Solution) -> Solution:
        """
        Variable Neighborhood Descent modulaire (Partie 6.2: amélioration locale).
        Ordre hiérarchique: TWO_OPT → RELOCATE → SWAP (cyclique évité par break imp).
        While improved: gen k=5 cands (modulaire generate_candidates), take first delta<0 or aspiration
        (feasible & cost < best global). Update local via copy + evaluate.
        Pré: Init_candidate (post-repair ; évalué).
        Post: Local opt (amélioré ou inchangé).
        Complexité: O(iters * k * |routes|^2) ; iters <=|types|=3, k=5 (scalable ~O(n^2)).
        Uniformité Notebook: Aspiration globale best (feas prefer) ; ordre efficacité (intra→inter).
        Note: Copy non-mut ; evaluate full post-move (incrémental possible mais simple).
        """
        local_solution = init_candidate.copy()  # Start local search
        improved = True
        order = [MoveType.TWO_OPT, MoveType.RELOCATE, MoveType.SWAP]  # Hiérarchie Partie 5
        while improved:
            improved = False
            for mt in order:
                candidates = generate_candidates(local_solution, self.instance, mt, k=5)  # Top 5 modulaires
                for cand_sol, delta in candidates:
                    # Aspiration: feasible & meilleur global (outrepasse delta>0)
                    aspiration = (cand_sol.feasible and cand_sol.cost < self.best_solution.cost)
                    if delta < 0 or aspiration:  # Améliore local ou aspir
                        local_solution = cand_sol.copy()  # Update local
                        evaluate_solution(local_solution, self.instance)  # Re-éval cohérence
                        improved = True
                        break  # Next type (descent)
                if improved:
                    break  # Next while (re-scan ordre si imp)
        return local_solution  # Local opt

    def shake_solution(self) -> Solution:
        """
        Perturbation diversification (Partie 6.3: si stagnation >= NO_IMPROVEMENT_LIMIT=50).
        Applique num=2..5 relocates random (gen k=1) ; reset no_improvement=0.
        Pré: Current stagn (no_improv >=50).
        Post: Shaken (perturbé ; évalué post) ou copy si <limit (no shake).
        Complexité: O(num * |routes|^2) gen/apply ; num~3 (faible).
        Uniformité Notebook: Relocate pour diversité inter-routes ; log activation.
        Note: Si <limit, return copy inchangé ; éval post chaque relocate.
        """
        if self.no_improvement < NO_IMPROVEMENT_LIMIT:
            return self.current_solution.copy()  # Pas de shake
        num_shakes = random.randint(2, 5)  # Intensité random
        shaken = self.current_solution.copy()
        print(f"🔄 Shake activé (stagnation {self.no_improvement}): {num_shakes} relocates random.")
        for _ in range(num_shakes):
            candidates = generate_candidates(shaken, self.instance, MoveType.RELOCATE, k=1)  # 1 random
            if candidates:
                shaken = candidates[0][0].copy()  # Applique best (seul)
                evaluate_solution(shaken, self.instance)  # Update post-shake
        self.no_improvement = 0  # Reset compteur
        return shaken

    def adapt_weights(self):
        """
        Adaptation dynamique opérateurs (Partie 6: EMA glissante window=10).
        Pour chaque op: avg=mean(hist[-10]), weight = (1-α)old + α (avg*10) ; trim hist à -10.
        Sym destroy/repair (même hist ; récompense scores 0/1 → weights ~0-10).
        Pré: Hist accumulés iters (append destroy/repair).
        Post: Poids updatés (favor ops performants: high score → high weight roulette).
        Complexité: O(|MoveType| * window) = O(1) (3*10).
        Uniformité Notebook: α=0.1 ; *10 scale (scores binaires → poids significatifs).
        Note: Si len(hist)<10, no update (init phases).
        """
        window_size = 10
        for op in MoveType:  # Par type (TWO/RELOC/SWAP)
            history = self.weights_history[op]  # List scores
            if len(history) >= window_size:
                avg_score = np.mean(history[-window_size:])  # Moyenne glissante
                # Update destroy/repair sym (adaptable asym)
                self.destroy_operators[op] = ((1 - WEIGHT_UPDATE) * self.destroy_operators[op] +
                                              WEIGHT_UPDATE * (avg_score * 10))
                self.repair_operators[op] = ((1 - WEIGHT_UPDATE) * self.repair_operators[op] +
                                             WEIGHT_UPDATE * (avg_score * 10))
                # Trim mémoire (last window)
                self.weights_history[op] = history[-window_size:]

    def run_iteration(self) -> bool:
        """
        Itération ALNS complète (Partie 6 flux: destroy → repair → accept? → VND local opt → update current/best/penalty/no_improv → shake si stagn → adapt).
        Retour: True si best améliorée (feasible & cost < best ; log conv).
        Pré: Current/best évalués ; itér <MAX_ITER=2000.
        Post: Current updaté ; best si imp ; penalty/no_improv adj ; weights adapt.
        Complexité: O(n^2) dominant (repair/gen VND) ; scalable.
        Uniformité Notebook: Penalty *0.99 si imp feas (favor feasible), *1.1 every 10 no_improv (pousse feas) ;
        shake >=50 ; adapt fin iter.
        Erreurs: Logs shake/penalty adj ; assume Kmax infini si None.
        """
        # 1. Destruction partielle
        destroyed = self.destroy()
        # 2. Reconstruction
        candidate = self.repair(destroyed)
        best_improved = False  # Flag retour
        # 3. Acceptation candidate
        if self.accept_solution(candidate):
            # 4. Amélioration locale VND
            improved_local = self.vnd_local_search(candidate)
            evaluate_solution(improved_local, self.instance)  # Cohérence post-VND
            self.current_solution = improved_local  # Update current
            # 5. Update best/penalty/no_improv
            if (improved_local.feasible and
                improved_local.cost < self.best_solution.cost):
                self.best_solution = improved_local.copy()  # Global best
                self.no_improvement = 0  # Reset stagn
                self.penalty *= 0.99  # Réduit penalty (plus feasible)
                best_improved = True  # Imp best
            else:
                self.no_improvement += 1  # Stagn
                if self.no_improvement % 10 == 0:
                    self.penalty *= 1.1  # Augmente (tolère moins infaisables)
        # 6. Shake si stagnation
        if self.no_improvement >= NO_IMPROVEMENT_LIMIT:
            self.current_solution = self.shake_solution()
        # 7. Adaptation poids
        self.adapt_weights()
        return best_improved  # Pour trace conv (imp best)


# --- [BLOC DE TEST] ---

if __name__ == "__main__":
    """
    Section de tests exécutable via : python -m src.alns
    (Test d'intégration majeur)
    """
    print("🚀 Lancement des tests d'intégration pour src/alns.py...")
    import sys
    import math

    # --- Dépendances de test ---
    # Ce test a besoin de TOUS les modules corrigés
    # Quand on lance avec "python -m src.alns",
    # la racine (Projet_vrp) est dans le path.
    # Les imports doivent être absolus depuis la racine.
    try:
        from src.contracts import Instance, Solution
        from src.evaluation import evaluate_solution
        from src.initial_solution import build_clarke_wright_solution
        # (alns.py importe déjà neighborhoods et evaluation en relatif)
    except ImportError as e:
        print(f"❌ ÉCHEC: Impossible d'importer les dépendances ({e}).")
        print("   Assurez-vous que contracts, evaluation, et initial_solution sont corrigés.")
        sys.exit(1)

    # --- Données de test ---
    DM_test = [
        [0.0, 10.0, 10.0, 100.0, 100.0], # 0
        [10.0, 0.0, 2.0, 100.0, 100.0], # 1
        [10.0, 2.0, 0.0, 100.0, 100.0], # 2
        [100.0, 100.0, 100.0, 0.0, 5.0],  # 3
        [100.0, 100.0, 100.0, 5.0, 0.0]   # 4
    ]
    tiny_instance = Instance(
        name="test_alns_engine",
        distance_matrix=DM_test,
        demand=[0, 1, 1, 1, 1], # 4 clients
        capacity=3, # C&W devrait trouver 2 routes
        Kmax=4
    )
    
    # On utilise C&W pour une solution de départ réaliste
    initial_solution = build_clarke_wright_solution(tiny_instance)
    cost_initial = initial_solution.cost
    print(f"Solution initiale (C&W) générée. Coût: {cost_initial:.2f}") # Attendu 44.0

    # --- Test 1: Initialisation ---
    print("\n--- Test 1: Initialisation ALNS ---")
    try:
        alns = ALNS(tiny_instance, initial_solution)
        print(f"✅ ALNS initialisé.")
        assert math.isclose(alns.best_solution.cost, cost_initial), "Le coût initial n'a pas été copié."
    except Exception as e:
        print(f"❌ ÉCHEC: L'initialisation de ALNS a planté: {e}")
        sys.exit(1)

    # --- Test 2: Exécution d'itérations ---
    print("\n--- Test 2: Lancement de 10 itérations ALNS ---")
    try:
        for i in range(10):
            print(f"  Iter {i+1}/10...")
            alns.run_iteration()
        
        print("\n✅ 10 itérations terminées sans crash.")
    except Exception as e:
        print(f"❌ ÉCHEC: alns.run_iteration() a planté: {e}")
        print("   Causes probables : 'generate_candidates' buggé,")
        print("   ou 'neighborhoods.py' n'est pas compatible (indices/faisabilité).")
        sys.exit(1)
        
    # --- Vérification Finale ---
    final_cost = alns.best_solution.cost
    print(f"\nCoût initial : {cost_initial:.2f}")
    print(f"Coût final   : {final_cost:.2f}")
    
    assert final_cost <= cost_initial, "L'ALNS a dégradé la solution (ne devrait pas arriver)"
    if final_cost < cost_initial:
        print("   (Amélioration trouvée !)")

    print("\n🎉 Tous les tests d'intégration ALNS ont réussi!")

    