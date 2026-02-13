from llm_social_simulation.simulation.analytics import cooperation_rate_per_round
from llm_social_simulation.simulation.engine import SimulationEngine
from llm_social_simulation.simulation.gameworld import Gameworld, PayoffMatrix


class AlwaysC:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def decide(self, obs):
        return {j: "C" for j in obs["neighbors"]}


class AlwaysD:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def decide(self, obs):
        return {j: "D" for j in obs["neighbors"]}


def make_ring_graph(n: int, k_each_side: int = 1):
    g = {i: set() for i in range(n)}
    for i in range(n):
        for d in range(1, k_each_side + 1):
            g[i].add((i + d) % n)
            g[i].add((i - d) % n)
    return {i: sorted(neis) for i, neis in g.items()}


def main():
    n = 10
    graph = make_ring_graph(n=n, k_each_side=1)
    payoff = PayoffMatrix(T=5, R=3, P=1, S=0)

    gw = Gameworld(graph=graph, payoff=payoff)
    agents = [AlwaysC(i) if i < 5 else AlwaysD(i) for i in range(n)]

    engine = SimulationEngine(gameworld=gw, agents=agents)
    history = engine.run(rounds=10)

    print("OK: ran 10 rounds")
    print("Final wealth:", history[-1].wealth)
    print("Round 0 delta:", history[0].delta_payoff)
    print("Round 0 actions for agent 0:", history[0].actions[0])

    rates = cooperation_rate_per_round(history)
    print("Cooperation rate per round:", rates)


def test_smoke_runs_and_updates_wealth():
    n = 10
    graph = make_ring_graph(n=n, k_each_side=1)
    payoff = PayoffMatrix(T=5, R=3, P=1, S=0)

    gw = Gameworld(graph=graph, payoff=payoff)
    agents = [AlwaysC(i) if i < 5 else AlwaysD(i) for i in range(n)]

    engine = SimulationEngine(gameworld=gw, agents=agents)
    history = engine.run(rounds=5)

    assert len(history) == 5
    assert any(v != 0 for v in history[-1].wealth.values())

    # optional: sanity check analytics runs and returns correct length
    rates = cooperation_rate_per_round(history)
    assert len(rates) == 5
    assert all(0.0 <= r <= 1.0 for r in rates)


if __name__ == "__main__":
    main()
