from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import copy

Action = str  # "C" or "D"


@dataclass(frozen=True)
class PayoffMatrix:
    """
    Prisoner's Dilemma payoff:
    - (C, C) -> (R, R)
    - (D, D) -> (P, P)
    - (D, C) -> (T, S)
    - (C, D) -> (S, T)
    Must satisfy: T > R > P > S
    """

    T: int = 5
    R: int = 3
    P: int = 1
    S: int = 0

    def payoff(self, a_i: Action, a_j: Action) -> Tuple[int, int]:
        if a_i == "C" and a_j == "C":
            return self.R, self.R
        if a_i == "D" and a_j == "D":
            return self.P, self.P
        if a_i == "D" and a_j == "C":
            return self.T, self.S
        if a_i == "C" and a_j == "D":
            return self.S, self.T
        raise ValueError(f"Invalid actions: {a_i}, {a_j}")


@dataclass
class TickResult:
    round: int
    actions: Dict[int, Dict[int, Action]]  # i -> (neighbor -> action)
    delta_payoff: Dict[int, int]  # i -> payoff gained this round
    wealth: Dict[int, int]  # snapshot after update


class Gameworld:
    """
    Single source of truth:
    - graph: adjacency list (undirected)
    - wealth: cumulative payoff per agent
    - last_actions: directed record of last round actions (i, j) -> action i took to j
    - t: current round index
    """

    def __init__(self, graph: Dict[int, List[int]], payoff: PayoffMatrix | None = None):
        self.graph = graph
        self.payoff = payoff or PayoffMatrix()
        self.wealth: Dict[int, int] = {i: 0 for i in graph.keys()}
        self.last_actions: Dict[Tuple[int, int], Action] = {}
        self.t: int = 0

    def neighbors(self, i: int) -> List[int]:
        return self.graph[i]

    def get_observation(self, i: int) -> Dict[str, Any]:
        """
        Observation should be local + minimal.
        For MVP: only include what neighbors did to me last round, my wealth, round.
        """
        neis = self.neighbors(i)
        last_from_neighbors = {j: self.last_actions.get((j, i), "C") for j in neis}

        return {
            "self_id": i,
            "neighbors": neis,
            "last_actions_from_neighbors": last_from_neighbors,
            "self_wealth": self.wealth[i],
            "round": self.t,
        }

    def apply_actions(self, actions: Dict[int, Dict[int, Action]]) -> TickResult:
        """
        actions[i][j] is the action agent i takes toward neighbor j in this round.

        We compute payoff on each undirected edge once (i < j) to avoid double counting.
        """

        # --- validate ---
        for i in self.graph.keys():
            if i not in actions:
                raise ValueError(f"Missing actions for agent {i}")
            for j in self.neighbors(i):
                if j not in actions[i]:
                    raise ValueError(f"Agent {i} missing action toward neighbor {j}")
                if actions[i][j] not in ("C", "D"):
                    raise ValueError(f"Invalid action {actions[i][j]} from {i} to {j}")

        delta = {i: 0 for i in self.graph.keys()}

        # --- payoff update (undirected edges counted once) ---
        for i in self.graph.keys():
            for j in self.neighbors(i):
                if i < j:
                    a_ij = actions[i][j]
                    a_ji = actions[j][i]
                    pi, pj = self.payoff.payoff(a_ij, a_ji)
                    delta[i] += pi
                    delta[j] += pj

        # --- update wealth ---
        for i in self.graph.keys():
            self.wealth[i] += delta[i]

        # --- update last_actions for next round (directed store) ---
        new_last: Dict[Tuple[int, int], Action] = {}
        for i in self.graph.keys():
            for j in self.neighbors(i):
                new_last[(i, j)] = actions[i][j]
        self.last_actions = new_last

        tick = TickResult(
            round=self.t,
            actions=copy.deepcopy(actions),
            delta_payoff=delta,
            wealth=copy.deepcopy(self.wealth),
        )
        self.t += 1
        return tick
