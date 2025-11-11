#Olivier
# on suppose la classe instance deja definit ou minimum declaree car elle est utilise dans cette section.
class ALNS:
    """
    Classe ALNS (Partie 6: métaheuristique adaptive ; destroy/repair itératif avec adaptation poids).
    Explore vaste voisinage via destruction partielle + reconstruction, VND local, shake diversification.
    Conforme notebook Section 3: robuste pour contraintes complexes (TW/cap/N_max/K_max).
    Pré: instance + initial_sol ; post: best_solution améliorée après run.
    
    Attributs:
        instance (Instance): Données/problème.
        destroy_operators/repair_operators (Dict[MoveType,float]): Poids (init 1.0 ; adapt via scores).
        current_solution/best_solution (Solution): Courante/meilleure (update si < best).
        weights_history (Dict[MoveType,List[float]]): Scores récents (0/1 feasible/amélioration).
        penalty (float): Dynamique (augmente stagnation, diminue succès ; gère infaisable).
        no_improvement (int): Compteur (pour shake si >= LIMIT).
    
    Méthodes: run_iteration (full loop), destroy/repair (opérateurs), accept (critère), VND (local),
    shake (diversif), adapt_weights (update alpha*moyenne).
    Complexité itération: O(n^2) approx (destroy O(n*fraction), repair O(n^2), VND O(k*n)).
    """
    def __init__(self, instance: Instance, initial_solution: Solution):
        """
        Init ALNS (Partie 6: setup). Copie initiale en current/best ; poids uniformes ; penalty init.
        Pré: initial feasible/partielle ; post: prête pour itérations.
        Complexité: O(taille initial) - copies.
        """
        self.instance = instance  # Réf problème
        self.destroy_operators = {mt: 1.0 for mt in MoveType}  # Poids destroy (roulette)
        self.repair_operators = {mt: 1.0 for mt in MoveType}  # Poids repair
        self.current_solution = initial_solution.copy()  # Courante (mutable)
        self.best_solution = initial_solution.copy()  # Meilleure (backup)
        self.weights_history = {mt: [] for mt in MoveType}  # Historique scores (append 0/1)
        self.penalty = PENALTY_FACTOR  # Init pénalité (ajustable)
        self.no_improvement = 0  # Compteur stagnation

    def destroy(self, fraction: float = SEGMENT_SIZE) -> Solution:
        """
        Destruction (Partie 6: partielle ; remove ~fraction*n clients aléatoirement de routes).
        Sélection opérateur via roulette (poids cum ; random.uniform(0,total_weight)).
        Simule relocate inverse (remove set random clients ; pop route si vide [0,0]).
        Score: 1 si post feasible (rare). Complexité: O(n*fraction) - sample + remove loops.
        """
        # Roulette pour opérateur (adaptatif ; cf. Partie 6)
        total_weight = sum(self.destroy_operators.values())  # Somme poids
        rand = random.uniform(0, total_weight)  # Tirage [0, total)
        cum_weight = 0.0  # Cumul progressif
        selected_op = None  # Op sélectionné
        for op, weight in self.destroy_operators.items():  # Parcours dict
            cum_weight += weight  # + poids courant
            if rand <= cum_weight:  # Dans segment ? → sélection
                selected_op = op
                break  # Arrêt
        
        # Destruction: sample clients à remove
        destroyed_sol = self.current_solution.copy()  # Copie pour mod
        num_to_destroy = int(fraction * self.instance.n)  # Nb ~ fraction*n
        all_clients = list(self.instance.Vc)  # Tous clients
        to_destroy = set(random.sample(all_clients, min(num_to_destroy, len(all_clients))))  # Set aléatoire
        
        for client in to_destroy:  # Boucle remove
            removed = False  # Flag
            for route_idx in range(len(destroyed_sol.routes)):  # Parcours routes
                route = destroyed_sol.routes[route_idx]  # Courante
                if client in route.sequence[1:-1]:  # Client dans internals ?
                    route.sequence.remove(client)  # Remove (O(len) list)
                    removed = True
                    # Vide ? [0,0] → pop (inutile)
                    if len(route.sequence) == 2 and route.sequence == [0, 0]:
                        destroyed_sol.routes.pop(route_idx)  # Supprime (index ajusté)
                    break  # Client removed
            if not removed:
                print(f"Attention: Client {client} non trouvé pour destruction.")  # Log rare
        
        destroyed_sol.compute_total_cost(self.instance)  # Update F/feasible post-remove
        # Score historique (1 feasible post-destroy ; pour adapt)
        score = 1.0 if destroyed_sol.is_feasible else 0.0
        self.weights_history[selected_op].append(score)  # Append list
        
        return destroyed_sol  # Partielle (unassigned)

    def repair(self, destroyed_sol: Solution) -> Solution:
        """
        Reconstruction (Partie 6: réinsère unassigned via glouton time-aware ; tri e_i).
        Sélection opérateur roulette (comme destroy). Meilleure pos: min delta_approx (arcs affectés)
        si feasible (is_feasible_route). Nouvelle route si possible. Score: 1 si feasible ET < current.
        Complexité: O(|unassigned| * |routes| * avg_len) ~ O(n^2) - scan pos/feasible.
        """
        # Roulette repair (adaptatif)
        total_weight = sum(self.repair_operators.values())
        rand = random.uniform(0, total_weight)
        cum_weight = 0.0
        selected_op = None
        for op, weight in self.repair_operators.items():
            cum_weight += weight
            if rand <= cum_weight:
                selected_op = op
                break
        
        # Unassigned (set diff Vc - covered)
        unassigned = set(self.instance.Vc) - destroyed_sol.covered_clients
        repaired = destroyed_sol.copy()  # Base
        
        for client in sorted(unassigned, key=lambda i: self.instance.e[i]):  # Tri time-aware (e_i croissant)
            # Meilleure insertion (min delta approx)
            best_delta = float('inf')  # Init max
            best_r_idx, best_pos = -1, -1  # Meilleurs
            for r_idx, route in enumerate(repaired.routes):  # Parcours routes
                for pos in range(1, len(route.sequence)):  # Pos (1..len-1)
                    # Temp seq insert
                    temp_seq = route.sequence[:pos] + [client] + route.sequence[pos:]
                    temp_r = Route(sequence=temp_seq)
                    if self.instance.is_feasible_route(temp_r):  # Vérif (TW/Q/etc.)
                        # Delta approx (Partie 4: incrémental ; arcs affectés seulement)
                        prev_n = route.sequence[pos-1]  # Précédent pos
                        next_n = route.sequence[pos] if pos < len(route.sequence)-1 else 0  # Suivant (0 si fin)
                        old_arc = self.instance.c[prev_n][next_n]  # Ancien arc coût
                        new_arcs = self.instance.c[prev_n][client] + self.instance.c[client][next_n]  # Nouveaux 2 arcs
                        delta_approx = new_arcs - old_arc  # Gain (négatif=bon ; ignore TW/charge via feasible)
                        if delta_approx < best_delta:  # Mieux ?
                            best_delta = delta_approx
                            best_r_idx, best_pos = r_idx, pos  # Update
            
            # Insertion si trouvée
            if best_r_idx != -1:
                repaired.routes[best_r_idx].sequence.insert(best_pos, client)  # In-place
            else:  # Nouvelle route
                if len(repaired.routes) < self.instance.K_max:
                    new_r = Route(sequence=[0, client, 0])
                    if self.instance.is_feasible_route(new_r):
                        repaired.routes.append(new_r)  # Ajout
                    else:
                        print(f"Client {client} non réinséré (nouvelle route infaisable).")  # Log
                else:
                    print(f"Client {client} non réinséré (K_max atteint).")  # Log
        
        repaired.compute_total_cost(self.instance)  # Update global
        # Score (1 si feasible ET amélioration ; pour adapt)
        score = 1.0 if (repaired.is_feasible and repaired.total_cost < self.current_solution.total_cost) else 0.0
        self.weights_history[selected_op].append(score)
        
        return repaired  # Réparée

    def accept_solution(self, new_sol: Solution) -> bool:
        """
        Acceptation (Partie 6: critère). Règles: Toujours si feasible ET < current ; sinon pénalité
        si !feasible (cost += penalty), accept si < current ; sinon Metropolis exp(-delta/T) T=100.
        Pré: new_sol candidate ; post: True si accept (update current).
        Complexité: O(1) - maths simples.
        """
        if new_sol.is_feasible and new_sol.total_cost < self.current_solution.total_cost:
            return True  # Amélioration stricte (feasible)
        
        # Pénalité si infaisable (Partie 6: gère temporaire)
        candidate_cost = new_sol.total_cost
        if not new_sol.is_feasible:
            candidate_cost += self.penalty  # + dynamique
        
        if candidate_cost < self.current_solution.total_cost:
            return True  # Amélioration pénalisée
        
        # Probabiliste diversification (Metropolis-like ; T=100 fixe)
        delta = candidate_cost - self.current_solution.total_cost  # Positif = pire
        prob = np.exp(-delta / 100.0)  # Prob accept (décroît avec delta)
        return random.random() < prob  # Tirage < prob ?

    def vnd_local_search(self, init_sol: Solution) -> Solution:
        """
        VND (Partie 6: descente voisinage variable ; séquentiel two→relocate→swap jusqu'à local opt).
        Accepte delta<0 ou aspiration (!feasible mais < best). Évite cycles: 1 move/type, break si improved.
        Pré: init_sol ; post: local_sol améliorée. Complexité: O(iters * k * n) ~ O(n^2) (k=5 petit).
        """
        local_sol = init_sol.copy()  # Base
        improved = True  # Flag loop
        neighborhood_order = [MoveType.TWO_OPT, MoveType.RELOCATE, MoveType.SWAP]  # Ordre séquentiel
        
        while improved:  # Tant amélioration
            improved = False  # Reset
            for mt in neighborhood_order:  # Parcours voisinages
                candidates = generate_candidates(local_sol, self.instance, mt, k=5)  # 5 aléatoires
                for cand_sol, delta in candidates:  # Meilleurs first (trié)
                    # Aspiration: même infaisable si < best global (Partie 6)
                    aspiration = (cand_sol.is_feasible and cand_sol.total_cost < self.best_solution.total_cost)
                    if delta < 0 or aspiration:  # Accepte
                        local_sol = cand_sol.copy()  # Update
                        local_sol.compute_total_cost(self.instance)  # Recompute
                        improved = True  # Flag
                        break  # 1 move/type (évite cycles)
                if improved:  # Si move, break pour new tour
                    break  # Nouveau while
        
        return local_sol  # Local opt

    def shake_solution(self) -> Solution:
        """
        Shake diversification (Partie 6: si stagnation >= LIMIT, 2-5 relocates random pour perturber).
        Reset no_improvement. Pré: stagnation ; post: shaken diversifiée.
        Complexité: O(num_shakes * |routes|) - relocates.
        """
        if self.no_improvement < NO_IMPROVEMENT_LIMIT:  # Pas stagnation
            return self.current_solution.copy()  # No-op
        
        num_shakes = random.randint(2, 5)  # Nb perturbations (2-5)
        shaken = self.current_solution.copy()  # Base
        print(f"Shake: {num_shakes} relocates random pour diversification.")  # Log
        
        for _ in range(num_shakes):  # Boucle shakes
            candidates = generate_candidates(shaken, self.instance, MoveType.RELOCATE, k=1)  # 1 random relocate
            if candidates:  # Si valide
                shaken = candidates[0][0].copy()  # Applique premier
                shaken.compute_total_cost(self.instance)  # Update
        
        self.no_improvement = 0  # Reset compteur
        return shaken  # Diversifiée

    def adapt_weights(self):
        """
        Adaptation dynamique (Partie 6: update poids = (1-alpha)*old + alpha*mean(scores[-10:])*boost).
        Boost=10 pour sensibilité ; trim history à 10 (fenêtre glissante). Alpha=0.1.
        Pré: history appends ; post: poids mis à jour (favorise bons opérateurs).
        Complexité: O(3*10) = O(1) - |MoveType|=3.
        """
        recent_window = 10  # Fenêtre moyenne (dernières 10 uses)
        for op in MoveType:  # Par type
            history = self.weights_history[op]  # List scores
            if len(history) >= recent_window:  # Suffisant ?
                avg_score = np.mean(history[-recent_window:])  # Moyenne récente (0/1)
                # Update destroy (formule EMA-like)
                self.destroy_operators[op] = (
                    (1 - WEIGHT_UPDATE) * self.destroy_operators[op] +  # Héritage old
                    WEIGHT_UPDATE * avg_score * 10  # + boost mean (sensibilité)
                )
                # Même pour repair
                self.repair_operators[op] = (
                    (1 - WEIGHT_UPDATE) * self.repair_operators[op] +
                    WEIGHT_UPDATE * avg_score * 10
                )
                # Trim (garde dernières 10)
                self.weights_history[op] = history[-recent_window:]

    def run_iteration(self) -> bool:
        """
        Itération ALNS complète (Partie 6: destroy→repair→accept→VND→update→shake?→adapt).
        Retourne True si best améliorée (réduit penalty). Complexité: O(n^2).
        """
        # 1. Destroy partiel
        destroyed = self.destroy()  # Remove fraction
        
        # 2. Repair
        candidate = self.repair(destroyed)  # Réinsère
        
        # 3. Accept + VND si OK
        if self.accept_solution(candidate):  # Critère
            improved_sol = self.vnd_local_search(candidate)  # Local improve
            improved_sol.compute_total_cost(self.instance)  # Update
            self.current_solution = improved_sol  # Set courant
            
            # 4. Update best (seulement feasible < best ; Partie 7)
            if (improved_sol.is_feasible and
                improved_sol.total_cost < self.best_solution.total_cost):
                self.best_solution = improved_sol.copy()  # Backup
                self.no_improvement = 0  # Reset
                self.penalty *= 0.99  # Réduit pénalité (plus de feasible)
                return True  # Amélioration
        
            else:
                self.no_improvement += 1  # + stagnation
                if self.no_improvement % 10 == 0:  # Tous 10
                    self.penalty *= 1.1  # Augmente (pousse feasible)
        
        # 5. Shake si stagnation extrême
        if self.no_improvement >= NO_IMPROVEMENT_LIMIT:
            self.current_solution = self.shake_solution()  # Diversif
        
        # 6. Adapt (poids)
        self.adapt_weights()
        
        return False  # Pas amélioration
