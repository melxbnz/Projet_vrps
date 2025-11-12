#Olivier
# on suppose la classe instance deja definit ou minimum declaree car elle est utilise dans cette section.
import numpy as np
import random
from dataclass import dataclass
from copy import deepcopy
from enum import Enum
from typing import List, Tuple, Optional, Dict
# Importation des modules artificiels 
from .contracts import Instance, Solution, Route 
from .evaluation import evaluate_solution, check_feasibility, compute_route_cost, delta_cost_two_opt



#=========================================================================================================
# Ma cCLASS ALNS : ADAPTATIVE LARGE NEIGHBORHOOD SEARCH.
#===========================================================================================================
class ALNS:
    """
    ALNS adaptative (Partie 6) ; utilise evaluate_solution/check_feasibility.
    Attributs: destroy/repair_operators (poids), current/best (Solution), history (scores),
    penalty (dynamique), no_improvement (stagnation).
    Méthodes: destroy/repair (mods routes + éval), accept (sur cost/feasible), VND (candidats),
    shake (relocates), adapt (poids EMA), run_iteration (full).
    Complexité itér: O(n^2).
    """
    def __init__(self, instance: Instance, initial_solution: Solution):
        """
        Init (Partie 6) ; copie initial (Solution.copy()), poids=1.0, penalty=1000, no_improv=0.
        Complexité: O(taille initial).
        """
        self.instance = instance
        self.destroy_operators = {mt: 1.0 for mt in MoveType}
        self.repair_operators = {mt: 1.0 for mt in MoveType}
        self.current_solution = initial_solution.copy()
        self.best_solution = initial_solution.copy()
        self.weights_history = {mt: [] for mt in MoveType}
        self.penalty = PENALTY_FACTOR
        self.no_improvement = 0
        evaluate_solution(self.current_solution, instance)  # Ensure updated
        evaluate_solution(self.best_solution, instance)

    def destroy(self, fraction: float = SEGMENT_SIZE) -> Solution:
        """
        Destruction (Partie 6) ; roulette opérateur, remove fraction*|Vc| clients aléa.
        Pop [0,0] ; evaluate_solution post. Score history (1 si feasible post, rare).
        Complexité: O(n*fraction).
        """
        total_weight = sum(self.destroy_operators.values())
        rand = random.uniform(0, total_weight)
        cum = 0.0
        selected_op = None
        for op, w in self.destroy_operators.items():
            cum += w
            if rand <= cum:
                selected_op = op
                break
        destroyed = self.current_solution.copy()
        num_destroy = int(fraction * (len(self.instance.demand) - 1))
        clients = list(range(1, len(self.instance.demand)))
        to_remove = set(random.sample(clients, min(num_destroy, len(clients))))
        for client in to_remove:
            removed = False
            for r_idx in range(len(destroyed.routes)):
                route = destroyed.routes[r_idx]
                if client in route[1:-1]:
                    destroyed.routes[r_idx] = route[:route.index(client)] + route[route.index(client)+1:]
                    removed = True
                    if len(destroyed.routes[r_idx]) == 2 and destroyed.routes[r_idx] == [0, 0]:
                        destroyed.routes.pop(r_idx)
                    break
            if not removed:
                print(f"Client {client} non trouvé.")
        evaluate_solution(destroyed, self.instance)
        score = 1.0 if destroyed.feasible else 0.0
        self.weights_history[selected_op].append(score)
        return destroyed

    def repair(self, destroyed: Solution) -> Solution:
        """
        Repair (Partie 6) ; roulette opérateur, réinsère unassigned (tri ready_time).
        Meilleure pos: min delta approx si check_feasible ; nouvelle si <Kmax.
        evaluate_solution post. Score: 1 si feasible et cost < current.
        Complexité: O(n^2).
        """
        total_weight = sum(self.repair_operators.values())
        rand = random.uniform(0, total_weight)
        cum = 0.0
        selected_op = None
        for op, w in self.repair_operators.items():
            cum += w
            if rand <= cum:
                selected_op = op
                break
        # Unassigned: Vc - couverts (approx via routes)
        unassigned = set(range(1, len(self.instance.demand)))
        for route in destroyed.routes:
            unassigned -= set(route[1:-1])
        unassigned = sorted(unassigned, key=lambda i: self.instance.ready_time[i] if self.instance.ready_time else 0)
        repaired = destroyed.copy()
        for client in unassigned:
            best_delta = float('inf')
            best_r, best_pos = -1, -1
            for r_idx, route in enumerate(repaired.routes):
                for pos in range(1, len(route)):
                    temp = route[:pos] + [client] + route[pos:]
                    if check_feasibility(temp, self.instance):
                        prev = route[pos-1]
                        next_ = route[pos] if pos < len(route)-1 else 0
                        old = self.instance.distance_matrix[prev][next_]
                        new_ = self.instance.distance_matrix[prev][client] + self.instance.distance_matrix[client][next_]
                        d_approx = new_ - old
                        if d_approx < best_delta:
                            best_delta = d_approx
                            best_r, best_pos = r_idx, pos
            if best_r != -1:
                repaired.routes[best_r] = repaired.routes[best_r][:best_pos] + [client] + repaired.routes[best_r][best_pos:]
            else:
                if len(repaired.routes) < (self.instance.Kmax or len(self.instance.demand)-1):
                    new_r = [0, client, 0]
                    if check_feasibility(new_r, self.instance):
                        repaired.routes.append(new_r)
                    else:
                        print(f"Client {client} non réinséré (nouvelle infaisable).")
                else:
                    print(f"Client {client} non réinséré (Kmax).")
        evaluate_solution(repaired, self.instance)
        score = 1.0 if (repaired.feasible and repaired.cost < self.current_solution.cost) else 0.0
        self.weights_history[selected_op].append(score)
        return repaired

    def accept_solution(self, new_sol: Solution) -> bool:
        """
        Accept (Partie 6) ; hiérarchique: feasible & cost<current → true ; +penalty si !feasible & <current → true ;
        sinon Metropolis exp(-Δ/T) T=100.
        Complexité: O(1).
        """
        if new_sol.feasible and new_sol.cost < self.current_solution.cost:
            return True
        cand_cost = new_sol.cost
        if not new_sol.feasible:
            cand_cost += self.penalty
        if cand_cost < self.current_solution.cost:
            return True
        delta = cand_cost - self.current_solution.cost
        prob = np.exp(-delta / 100.0)
        return random.random() < prob

    def vnd_local_search(self, init_sol: Solution) -> Solution:
        """
        VND (Partie 6.2) ; ordre two→relocate→swap ; accepte delta<0 ou aspiration (feasible & cost<best).
        1 move/type ; while improved.
        Complexité: O(iters * k * |routes|).
        """
        local = init_sol.copy()
        improved = True
        order = [MoveType.TWO_OPT, MoveType.RELOCATE, MoveType.SWAP]
        while improved:
            improved = False
            for mt in order:
                cands = generate_candidates(local, self.instance, mt, k=5)
                for cand_sol, delta in cands:
                    aspiration = cand_sol.feasible and cand_sol.cost < self.best_solution.cost
                    if delta < 0 or aspiration:
                        local = cand_sol.copy()
                        evaluate_solution(local, self.instance)
                        improved = True
                        break
                if improved:
                    break
        return local

    def shake_solution(self) -> Solution:
        """
        Shake (Partie 6.3) ; si stagnation, 2-5 relocates random ; reset no_improv.
        Complexité: O(num * |routes|).
        """
        if self.no_improvement < NO_IMPROVEMENT_LIMIT:
            return self.current_solution.copy()
        num = random.randint(2, 5)
        shaken = self.current_solution.copy()
        print(f"Shake: {num} relocates.")
        for _ in range(num):
            cands = generate_candidates(shaken, self.instance, MoveType.RELOCATE, k=1)
            if cands:
                shaken = cands[0][0].copy()
                evaluate_solution(shaken, self.instance)
        self.no_improvement = 0
        return shaken

    def adapt_weights(self):
        """
        Adapt (Partie 6) ; EMA mean(last 10 scores)*10 ; trim 10.
        Complexité: O(1).
        """
        window = 10
        for op in MoveType:
            hist = self.weights_history[op]
            if len(hist) >= window:
                avg = np.mean(hist[-window:])
                self.destroy_operators[op] = (1 - WEIGHT_UPDATE) * self.destroy_operators[op] + WEIGHT_UPDATE * avg * 10
                self.repair_operators[op] = (1 - WEIGHT_UPDATE) * self.repair_operators[op] + WEIGHT_UPDATE * avg * 10
                self.weights_history[op] = hist[-window:]

    def run_iteration(self) -> bool:
        """
        Itér ALNS (Partie 6) ; destroy→repair→accept/VND→update→shake?→adapt.
        True si best améliorée.
        Complexité: O(n^2).
        """
        destroyed = self.destroy()
        candidate = self.repair(destroyed)
        if self.accept_solution(candidate):
            improved = self.vnd_local_search(candidate)
            evaluate_solution(improved, self.instance)
            self.current_solution = improved
            if improved.feasible and improved.cost < self.best_solution.cost:
                self.best_solution = improved.copy()
                self.no_improvement = 0
                self.penalty *= 0.99
                return True
            else:
                self.no_improvement += 1
                if self.no_improvement % 10 == 0:
                    self.penalty *= 1.1
        if self.no_improvement >= NO_IMPROVEMENT_LIMIT:
            self.current_solution = self.shake_solution()
        self.adapt_weights()
        return False



#=========================================================================================================
# FONCTION QUI GENERE K CANDIDATS 
#===========================================================================================================
def generate_candidates(solution: Solution, instance: Instance, move_type: MoveType, k: int = 10) -> List[Tuple[Solution, float]]:
    """
    Génère k candidats (Partie 5/6) ; tri delta croissant.
    Two-opt: delta O(1) ; relocate/swap: full evaluate.
    Complexité: O(k * |routes|^2).
    """
    candidates = []
    routes = solution.routes
    
    if move_type == MoveType.TWO_OPT:
        for _ in range(k):
            if len(routes) == 0 or any(len(r) < 4 for r in routes):
                continue
            r_idx = random.randint(0, len(routes)-1)
            route = routes[r_idx]
            seq_len = len(route)
            i = random.randint(1, seq_len-3)
            j = random.randint(i+1, seq_len-2)
            new_route, delta = two_opt_move(route, i, j, instance)
            if delta < float('inf'):
                new_sol = solution.copy()
                new_sol.routes[r_idx] = new_route
                evaluate_solution(new_sol, instance)  # Update pour cohérence
                candidates.append((new_sol, delta))
    
    elif move_type == MoveType.RELOCATE:
        for _ in range(k):
            from_r = random.randint(0, len(routes)-1)
            to_r = random.randint(0, len(routes)-1)
            if from_r == to_r and len(routes[from_r]) < 3:
                continue
            from_p = random.randint(1, len(routes[from_r])-2)
            to_p = random.randint(1, len(routes[to_r])-1)
            new_sol, delta = relocate_move(solution, from_r, from_p, to_r, to_p, instance)
            if delta < float('inf'):
                candidates.append((new_sol, delta))
    
    elif move_type == MoveType.SWAP:
        for _ in range(k):
            r1 = random.randint(0, len(routes)-1)
            r2 = random.randint(0, len(routes)-1)
            if r1 == r2 or len(routes[r1]) < 3 or len(routes[r2]) < 3:
                continue
            p1 = random.randint(1, len(routes[r1])-2)
            p2 = random.randint(1, len(routes[r2])-2)
            new_sol, delta = swap_move(solution, r1, p1, r2, p2, instance)
            if delta < float('inf'):
                candidates.append((new_sol, delta))
    
    return sorted(candidates, key=lambda x: x[1])[:k]
