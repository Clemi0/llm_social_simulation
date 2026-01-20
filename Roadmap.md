# LLM Social Simulation — Project Roadmap

This document describes the phased roadmap for the LLM Social Simulation project.
Each phase contains goals, deliverables, and acceptance criteria.

---

## Phase 0 — MVP: End-to-End Simulation Loop (Closed Loop + Tests)

**Status:** In progress (late stage)

### Goal

Build a minimal but complete simulation framework that can run Iterated Prisoner’s
Dilemma (IPD) on a fixed graph, produce analyzable outputs, and pass automated tests.

Phase 0 establishes a stable foundation or internal environment for all later phases.

---

### Core Capabilities

- Run multi-round IPD on a fixed graph
- Output:
  - Cooperation rate per round (time series)
  - Final wealth distribution
- Simulation-related tests pass (smoke tests)

---

### Deliverables (Minimal Set)

#### Agents
- `BaseAgent(agent_id)`
- `decide(observation) -> Dict[neighbor_id, action]`
- At least two strategies implemented:
  - Always Cooperate
  - Always Defect
  - (Preferred) Tit-for-Tat

#### Simulation Entry Point (optional)
- `run_mvp.py`
- Single official entry point for running the Phase 0 MVP

#### Tests
- `test_gameworld_smoke.py`
- `test_engine_dummy.py`
- All simulation tests pass under `pytest`

---

### Acceptance Criteria

- The following command runs successfully from the repo root:
  ```bash
  python -m llm_social_simulation.simulation.run_mvp
  ```
- Output includes: Cooperation rate over time and final wealth per agent
- pytest passes
  
