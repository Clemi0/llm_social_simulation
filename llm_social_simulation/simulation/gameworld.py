from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

Action = str  # "C" or "D"
AgentId = int


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

    def payoff(self, a_i: Action, a_j: Action) -> tuple[int, int]:
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
    actions: dict[int, dict[int, Action]]  # i -> (neighbor -> action)
    delta_payoff: dict[int, int]  # i -> payoff gained this round
    wealth: dict[int, int]  # snapshot after update


class Gameworld:
    """
    Single source of truth:
    - graph: adjacency list (undirected)
    - wealth: cumulative payoff per agent
    - last_actions: directed record of last round actions (i, j) -> action i took to j
    - t: current round index
    """

    def __init__(self, graph: dict[int, list[int]], payoff: PayoffMatrix | None = None):
        self.graph = graph
        self.payoff = payoff or PayoffMatrix()
        self.wealth: dict[int, int] = {i: 0 for i in graph.keys()}
        self.last_actions: dict[tuple[int, int], Action] = {}
        self.t: int = 0

    def neighbors(self, i: int) -> list[int]:
        return self.graph[i]

    def get_observation(self, i: int) -> dict[str, Any]:
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

    def apply_actions(self, actions: dict[int, dict[int, Action]]) -> TickResult:
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
        new_last: dict[tuple[int, int], Action] = {}
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


@dataclass(frozen=True)
class OpenResourcesConfig:
    """
    Contract-layer configuration for the Open Resources game.

    NOTE: this only establishes interface and default bounds. The full dynamics
    (allocation, governance pool recursion, regeneration) are intentionally not
    implemented in this step.
    """

    agent_ids: tuple[AgentId, ...]
    initial_resource: float = 100.0
    initial_pool: float = 0.0
    initial_wealth: float = 0.0
    max_harvest_per_step: float = 1_000_000.0


@dataclass(frozen=True)
class OpenResourcesAction:
    """Single-step action request by one agent."""

    harvest: float
    contribute: float


@dataclass(frozen=True)
class OpenResourcesObservation:
    """
    Per-agent observation payload for the Open Resources world.

    All fields are JSON-serializable to simplify logging and analytics.
    """

    self_id: AgentId
    t: int
    R: float
    P: float
    self_wealth: float
    known_agents: list[AgentId]
    info: dict[str, Any]


@dataclass(frozen=True)
class OpenResourcesTick:
    """
    Per-step result object for analytics and research.

    This is contract-only for now; many values may be placeholders until
    full dynamics are implemented.
    """

    t: int
    R_before: float
    R_after: float
    P_before: float
    P_after: float
    harvest_requested: dict[AgentId, float]
    harvest_actual: dict[AgentId, float]
    contribute: dict[AgentId, float]
    reward: dict[AgentId, float]
    wealth: dict[AgentId, float]
    clamped: dict[AgentId, dict[str, bool]]
    info: dict[str, Any]


@dataclass
class OpenResourcesState:
    """Internal snapshot container for mutable world state."""

    t: int
    R: float
    P: float
    wealth: dict[AgentId, float]


class OpenResourcesGameWorld:
    """
    Open Resources game contract adapter.

    This class intentionally implements only the interface expected by the
    simulation engine: `get_observation(agent_id)` and
    `apply_actions(actions) -> OpenResourcesTick`.
    """

    def __init__(self, config: OpenResourcesConfig):
        self.config = config
        self.state = OpenResourcesState(
            t=0,
            R=float(config.initial_resource),
            P=float(config.initial_pool),
            wealth={agent_id: float(config.initial_wealth) for agent_id in config.agent_ids},
        )

    def get_observation(self, agent_id: AgentId) -> OpenResourcesObservation:
        if agent_id not in self.state.wealth:
            raise ValueError(f"Unknown agent_id: {agent_id}")

        return OpenResourcesObservation(
            self_id=agent_id,
            t=self.state.t,
            R=self.state.R,
            P=self.state.P,
            self_wealth=self.state.wealth[agent_id],
            known_agents=list(self.config.agent_ids),
            info={"contract_only": True},
        )

    def apply_actions(self, actions: Mapping[AgentId, OpenResourcesAction]) -> OpenResourcesTick:
        """
        Validate and normalize actions, then emit a contract-complete tick.

        Full dynamics are deferred to the next implementation step.
        """
        harvest_requested: dict[AgentId, float] = {}
        harvest_actual: dict[AgentId, float] = {}
        contribute: dict[AgentId, float] = {}
        reward: dict[AgentId, float] = {}
        clamped: dict[AgentId, dict[str, bool]] = {}

        for agent_id in self.config.agent_ids:
            if agent_id not in actions:
                raise ValueError(f"Missing action for agent {agent_id}")

            action = actions[agent_id]
            if isinstance(action, OpenResourcesAction):
                requested_harvest = float(action.harvest)
                requested_contribute = float(action.contribute)
            else:
                requested_harvest = float(action["harvest"])
                requested_contribute = float(action["contribute"])

            wealth_now = self.state.wealth[agent_id]

            clamped_harvest = min(max(0.0, requested_harvest), self.config.max_harvest_per_step)
            clamped_contribute = min(max(0.0, requested_contribute), wealth_now)

            harvest_requested[agent_id] = clamped_harvest
            harvest_actual[agent_id] = (
                clamped_harvest  # placeholder until allocation is implemented
            )
            contribute[agent_id] = clamped_contribute
            reward[agent_id] = 0.0  # placeholder until governance reward logic is implemented
            clamped[agent_id] = {
                "harvest": clamped_harvest != requested_harvest,
                "contribute": clamped_contribute != requested_contribute,
            }

        tick = OpenResourcesTick(
            t=self.state.t,
            R_before=self.state.R,
            R_after=self.state.R,
            P_before=self.state.P,
            P_after=self.state.P,
            harvest_requested=harvest_requested,
            harvest_actual=harvest_actual,
            contribute=contribute,
            reward=reward,
            wealth=dict(self.state.wealth),
            clamped=clamped,
            info={"contract_only": True, "dynamics_implemented": False},
        )

        self.state.t += 1
        return tick
