# Open Renewable Resource Game Specification

*February 2026*

## Time, Agents, and Horizon

- **Time.** Decision steps are indexed by
  \[
  t \in \{0,1,\dots,T-1\}.
  \]
- **Agents.**
  \[
  \mathcal{N} = \{1,2,\dots,n\}.
  \]
- **Horizon.** The episode consists of \(T\) decision steps.
  The final action is taken at \(t=T-1\), after which the transition produces terminal state \(S_T\).
  The episode may terminate earlier if a collapse condition is triggered.

## State Space

### Base State (No Governance)

\[
S_t = (R_t, \mathbf{w}_t),
\]

where:

- \(R_t \in [0,K]\) is the renewable resource stock,
- \(\mathbf{w}_t = (w_1^t,\dots,w_n^t)\) is the wealth vector,
- \(w_i^t \in \mathbb{R}_{\ge 0}\) for all \(i \in \mathcal{N}\).

### Governance Extension (Optional)

If governance is enabled, the state includes a governance pool:

\[
S_t = (R_t, \mathbf{w}_t, P_t),
\]

where \(P_t \in \mathbb{R}_{\ge 0}\).

### Markov Property

Given joint action \(\mathbf{a}_t\),

\[
S_{t+1} = T(S_t, \mathbf{a}_t).
\]

The transition is deterministic unless environmental noise is explicitly introduced.

## Action Space

At time \(t\), each agent \(i \in \mathcal{N}\) selects

\[
a_i^t = (h_i^t, c_i^t),
\]

where

- \(h_i^t \in [0,h_{\max}]\) is the harvest request,
- \(c_i^t \in [0,w_i^t]\) is the governance contribution.

The joint action is

\[
\mathbf{a}_t = (a_1^t,\dots,a_n^t).
\]

### Action Validation

\[
h_i^t \leftarrow \min(\max(h_i^t,0), h_{\max}),
\]

\[
c_i^t \leftarrow \min(\max(c_i^t,0), w_i^t).
\]

## Allocation Rule (Proportional Scaling)

Total requested harvest:

\[
H_t^{\mathrm{req}} = \sum_{i=1}^n h_i^t.
\]

Scaling factor:

\[
\phi_t =
\begin{cases}
1, & \text{if } H_t^{\mathrm{req}} = 0, \\
\min\left(1, \dfrac{R_t}{H_t^{\mathrm{req}}}\right), & \text{otherwise.}
\end{cases}
\]

Realized harvest:

\[
\tilde{h}_i^t = \phi_t h_i^t.
\]

Total realized harvest:

\[
H_t = \sum_{i=1}^n \tilde{h}_i^t
= \min(R_t, H_t^{\mathrm{req}}).
\]

## Wealth Update

\[
w_i^{t+1} = w_i^t + \tilde{h}_i^t - c_i^t,
\qquad \forall i \in \mathcal{N}.
\]

## Governance (Optional)

Total contribution:

\[
C_t = \sum_{i=1}^{n} c_i^t.
\]

Two possible pool modes:

### (A) Per-step pool

\[
P_t = C_t.
\]

### (B) Accumulating pool with decay

\[
P_{t+1} = \rho P_t + C_t,
\qquad \rho \in [0,1].
\]

Governance bonus:

\[
G_t = \alpha P_t,
\qquad \alpha \ge 0.
\]

If governance is disabled, set \(c_i^t = 0\) for all \(i\), hence \(G_t = 0\).

## Resource Regeneration

### Logistic Regeneration (Default)

\[
R_{t+1} =
\min \left(
K,
\max \left(
0,
R_t - H_t
+ \gamma R_t \left(1 - \frac{R_t}{K}\right)
+ G_t
\right)
\right).
\]

### Linear Baseline

\[
R_{t+1} =
\min\left(
K,
\max\left(
0,
R_t - H_t + \gamma + G_t
\right)
\right).
\]

## Reward

Per-step reward:

\[
r_i^t = \tilde{h}_i^t - c_i^t.
\]

Optional discounted objective:

\[
\max \mathbb{E}\left[
\sum_{t=0}^{T-1} \delta^t r_i^t
\right],
\qquad \delta \in (0,1].
\]

## Observation Function

After step \(t\), agent \(i\) receives

\[
o_i^{t+1} = \Omega_i(S_{t+1}, \tilde{\mathbf{h}}_t, \mathbf{c}_t, \epsilon).
\]

Example (full observability):

\[
o_i^{t+1} =
(R_{t+1}, \mathbf{w}_{t+1}, \tilde{\mathbf{h}}_t, \mathbf{c}_t).
\]

Example (local observability with graph \(G=(\mathcal{N},E)\)):

Let \(\mathcal{N}(i)=\{j:(i,j)\in E\}\) be neighbors of \(i\).

\[
o_i^{t+1} =
\left(
R_{t+1},
w_i^{t+1},
\{w_j^{t+1}, \tilde{h}_j^t, c_j^t\}_{j\in \mathcal{N}(i)}
\right).
\]

Example (noisy stock):

\[
\hat{R}_{t+1} = R_{t+1} + \eta_{t+1},
\qquad
\eta_{t+1} \sim \mathcal{N}(0,\sigma_R^2).
\]

## Collapse and Termination

Collapse occurs if either

\[
R_t < R_{\mathrm{critical}},
\]

or

\[
R_t = 0 \text{ for } k \text{ consecutive steps}.
\]

If terminate-on-collapse is enabled, the episode ends immediately.

## Deterministic Step Order

Each step executes in the following order:

1. Receive raw actions.
2. Validate and clamp actions.
3. Compute proportional allocation.
4. Update wealth.
5. Update governance pool.
6. Compute governance bonus.
7. Update resource.
8. Compute rewards.
9. Build observations.
10. Check termination.
