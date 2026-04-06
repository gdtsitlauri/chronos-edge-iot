# CHRONOS: Causal Hypergraph-Regulated Orchestration of Networked Offloading with Spiking Intelligence

## A PhD Research Proposal

---

## 1. Title

**"CHRONOS: A Causal Hypergraph Framework for Joint Federated Learning, Task Scheduling, and Communication Optimization in Edge-IoT Systems via Spiking Multi-Agent Reinforcement Learning"**

---

## 2. Abstract

The proliferation of Internet-of-Things (IoT) devices at the network edge demands intelligent orchestration of computation, communication, and learning under stringent latency, energy, and bandwidth constraints. Existing approaches address federated learning, task offloading, and communication optimization largely in isolation, neglecting the higher-order causal dependencies that govern real-world edge ecosystems. We propose **CHRONOS** (Causal Hypergraph-Regulated Orchestration of Networked Offloading with Spiking intelligence), a novel algorithmic framework that unifies these dimensions through three foundational contributions. First, we introduce a **Causal Hypergraph State Representation (CHSR)** that models the edge-IoT system as a dynamic hypergraph where hyperedges encode causal interventional relationships among devices, tasks, channels, and models — moving beyond correlational graph representations. Second, we design a **Spiking Multi-Agent Reinforcement Learning (S-MARL)** architecture in which each edge agent employs spiking neural network policies for event-driven, energy-efficient decision-making, communicating via learned spike-coded messages over a shared causal hypergraph attention mechanism. Third, we formulate the joint optimization of model accuracy, end-to-end latency, energy consumption, and communication cost as a **Constrained Causal Multi-Objective Markov Game (CC-MOMG)**, solved through a novel causal counterfactual policy gradient method with provable convergence guarantees under non-IID, time-varying conditions. CHRONOS is validated on a Digital Twin simulation platform calibrated against real-world edge deployments. Theoretical analysis establishes sub-linear regret bounds and federated convergence rates under partial observability and non-stationary causal graphs. This work bridges causal representation learning, neuromorphic computing, and distributed optimization, offering a principled foundation for next-generation autonomous edge intelligence.

---

## 3. Problem Formulation

### 3.1 System Model

Consider an edge-IoT system comprising:

- A set of **edge nodes** $\mathcal{N} = \{1, 2, \dots, N\}$, each with heterogeneous compute capacity $c_i$, memory $m_i$, and energy budget $E_i^{\max}$.
- A set of **IoT devices** $\mathcal{D} = \{1, 2, \dots, D\}$ generating tasks and local data.
- A set of **tasks** $\mathcal{T}(t) = \{T_1(t), \dots, T_K(t)\}$ arriving dynamically at time $t$, each characterized by tuple $T_k = (\delta_k, \omega_k, d_k, \tau_k^{\max})$ where $\delta_k$ is input data size, $\omega_k$ is required computation (CPU cycles), $d_k$ is the data partition for learning, and $\tau_k^{\max}$ is the deadline.
- A set of **wireless channels** $\mathcal{C} = \{1, \dots, C\}$ with time-varying channel gains $h_{ij}^c(t)$ between device $i$ and edge node $j$ on channel $c$.
- A **federated learning process** over a global model $\mathbf{w}$ with local datasets $\{D_i\}_{i=1}^N$ that are **non-IID**.

### 3.2 Causal Hypergraph Representation

We model the system state as a **dynamic causal hypergraph**:

$$\mathcal{H}(t) = (\mathcal{V}(t), \mathcal{E}(t), \mathcal{W}(t), \mathcal{G}(t))$$

where:
- $\mathcal{V}(t) = \mathcal{N} \cup \mathcal{D} \cup \mathcal{T}(t) \cup \mathcal{C}$ is the vertex set (heterogeneous: nodes, devices, tasks, channels).
- $\mathcal{E}(t) \subseteq 2^{\mathcal{V}(t)}$ is the hyperedge set, where each hyperedge $e \in \mathcal{E}(t)$ connects an arbitrary subset of vertices involved in a **causal interaction** (e.g., a hyperedge $\{d_3, n_1, T_5, c_2\}$ encodes "device $d_3$ offloads task $T_5$ to node $n_1$ via channel $c_2$").
- $\mathcal{W}(t): \mathcal{E}(t) \rightarrow \mathbb{R}^+$ assigns causal strength weights.
- $\mathcal{G}(t)$ is a **structural causal model (SCM)** defined over the hypergraph: $\mathcal{G}(t) = (\mathbf{U}, \mathbf{V}_{\mathcal{H}}, \mathcal{F}, P(\mathbf{U}))$ where $\mathbf{V}_{\mathcal{H}}$ are endogenous variables associated with hypergraph elements, $\mathbf{U}$ are exogenous variables, $\mathcal{F}$ are structural equations, and $P(\mathbf{U})$ is the noise distribution.

**Key distinction from standard GNN-based representations:** Hyperedges capture multi-way causal relationships (not pairwise), and the SCM enables interventional reasoning (do-calculus) rather than purely observational correlation.

### 3.3 Decision Variables

At each decision epoch $t$, the system must jointly determine:

1. **Task offloading decisions** $\mathbf{x}(t) \in \{0,1\}^{K \times (N+1)}$: $x_{kj}(t) = 1$ if task $T_k$ is assigned to edge node $j$ (or $j=0$ for local execution).

2. **Resource allocation** $\mathbf{r}(t) \in \mathbb{R}_+^{K \times N}$: fraction of compute resource at node $j$ allocated to task $T_k$.

3. **Channel assignment and power control** $\mathbf{p}(t) \in \mathbb{R}_+^{D \times C}$: transmit power of device $i$ on channel $c$, and $\mathbf{a}(t) \in \{0,1\}^{D \times C}$: channel-device assignment.

4. **Federated learning parameters** $\boldsymbol{\theta}(t)$: local SGD steps $\kappa_i(t)$, aggregation weights $\alpha_i(t)$, compression ratio $\rho_i(t)$, and participation decisions $z_i(t) \in \{0,1\}$.

5. **Spiking encoding parameters** $\boldsymbol{\phi}(t)$: spike rate thresholds $\vartheta_i(t)$ and temporal coding windows $\Delta_i(t)$ governing each agent's neural encoding.

The composite action is $\mathbf{A}(t) = (\mathbf{x}(t), \mathbf{r}(t), \mathbf{p}(t), \mathbf{a}(t), \boldsymbol{\theta}(t), \boldsymbol{\phi}(t))$.

### 3.4 Objective Functions

**Objective 1 — Federated Model Accuracy (maximize):**

$$f_1(\mathbf{A}) = -\left[ F(\mathbf{w}^*) - F(\mathbf{w}^{(R)}) \right], \quad F(\mathbf{w}) = \sum_{i=1}^{N} \frac{|D_i|}{|D|} F_i(\mathbf{w})$$

where $F_i(\mathbf{w}) = \mathbb{E}_{\xi \sim D_i}[\ell(\mathbf{w}; \xi)]$ and $R$ is the number of global rounds.

**Objective 2 — End-to-End Latency (minimize):**

$$f_2(\mathbf{A}) = \max_{k \in \mathcal{T}(t)} \left[ \underbrace{\frac{\delta_k}{R_{ij}^c(t)}}_{\text{transmission}} + \underbrace{\frac{\omega_k}{c_j \cdot r_{kj}(t)}}_{\text{computation}} + \underbrace{\tau_k^{\text{queue}}(t)}_{\text{queuing}} \right]$$

where the achievable rate on channel $c$ is:

$$R_{ij}^c(t) = B \log_2\left(1 + \frac{p_i^c(t) |h_{ij}^c(t)|^2}{\sigma^2 + \sum_{i' \neq i} p_{i'}^c(t) |h_{i'j}^c(t)|^2}\right)$$

**Objective 3 — Energy Consumption (minimize):**

$$f_3(\mathbf{A}) = \sum_{i \in \mathcal{D}} \left[ \underbrace{\sum_c p_i^c(t) \cdot \frac{\delta_k}{R_{ij}^c(t)}}_{\text{communication energy}} + \underbrace{\kappa_i \cdot \gamma_i \cdot f_i^2 \cdot \omega_i^{\text{local}}}_{\text{local compute energy}} \right] + \sum_{j \in \mathcal{N}} \underbrace{E_j^{\text{comp}}(t)}_{\text{edge compute}}$$

where $\gamma_i$ is the effective capacitance coefficient and $f_i$ is CPU frequency.

**Objective 4 — Communication Cost (minimize):**

$$f_4(\mathbf{A}) = \sum_{i=1}^{N} z_i(t) \cdot \rho_i(t) \cdot \|\nabla F_i(\mathbf{w}_i^{(t)})\|_0 \cdot b$$

where $b$ is the bit-width per parameter and $\|\cdot\|_0$ counts non-zero gradient elements after sparsification.

### 3.5 Constraints

$$\text{C1 (Deadline):} \quad \forall k: \quad \tau_k^{\text{total}}(t) \leq \tau_k^{\max}$$

$$\text{C2 (Energy budget):} \quad \forall i: \quad \sum_{t=0}^{T} E_i(t) \leq E_i^{\max}$$

$$\text{C3 (Compute capacity):} \quad \forall j: \quad \sum_{k} r_{kj}(t) \leq 1$$

$$\text{C4 (Task assignment):} \quad \forall k: \quad \sum_{j=0}^{N} x_{kj}(t) = 1$$

$$\text{C5 (Power):} \quad \forall i: \quad \sum_c p_i^c(t) \leq P_i^{\max}$$

$$\text{C6 (Causality):} \quad \mathbf{A}(t) \in \mathcal{A}^{\text{causal}}(\mathcal{H}(t))$$

Constraint C6 is novel: it restricts actions to those consistent with the learned causal structure — an agent cannot assign a task to a node if the causal graph indicates the communication link is causally blocked.

### 3.6 Formal Multi-Objective Optimization

$$\min_{\pi_1, \dots, \pi_N} \quad \mathbf{F}(\boldsymbol{\pi}) = \left(-f_1, f_2, f_3, f_4\right)$$

$$\text{subject to} \quad \text{C1}–\text{C6}$$

$$\text{where} \quad \pi_i: \mathcal{S}_i \times \mathcal{H}(t) \rightarrow \Delta(\mathcal{A}_i), \quad \forall i \in \mathcal{N}$$

This is formalized as a **Constrained Causal Multi-Objective Markov Game (CC-MOMG)**:

$$\mathfrak{G} = \langle \mathcal{N}, \{\mathcal{S}_i\}, \{\mathcal{A}_i\}, \mathcal{H}, P, \{R_i^{(m)}\}_{m=1}^4, \gamma, \{C_j\}_{j=1}^6 \rangle$$

where $P: \mathcal{S} \times \mathcal{A} \times \mathcal{H} \rightarrow \Delta(\mathcal{S} \times \mathcal{H})$ is the causal transition kernel defined via the SCM, and $R_i^{(m)}$ is the reward for agent $i$ under objective $m$.

---

## 4. Proposed Algorithm: CHRONOS

### 4.1 Full Name

**CHRONOS**: **C**ausal **H**ypergraph-**R**egulated **O**rchestration of **N**etworked **O**ffloading with **S**piking Intelligence

### 4.2 Architecture Overview

CHRONOS consists of five tightly integrated modules:

```
+===================================================================+
|                        CHRONOS FRAMEWORK                          |
+===================================================================+
|                                                                   |
|  Module 1: CAUSAL HYPERGRAPH STATE ENCODER (CHSE)                |
|  ┌─────────────────────────────────────────────────────────┐      |
|  │  Dynamic Hypergraph Construction → Causal Discovery     │      |
|  │  → Hypergraph Attention Network → SCM-based Encoding    │      |
|  └─────────────────────────────────────────────────────────┘      |
|           ↓ Causal State Embedding z(t)                           |
|  Module 2: SPIKING POLICY NETWORK (SPN)                          |
|  ┌─────────────────────────────────────────────────────────┐      |
|  │  Spike Encoder → Leaky Integrate-and-Fire Layers        │      |
|  │  → Temporal Difference Spike Timing (TDST) Decoder      │      |
|  └─────────────────────────────────────────────────────────┘      |
|           ↓ Spike-coded Actions                                   |
|  Module 3: CAUSAL COUNTERFACTUAL POLICY GRADIENT (CCPG)          |
|  ┌─────────────────────────────────────────────────────────┐      |
|  │  Interventional Advantage Estimation → Counterfactual   │      |
|  │  Baseline → Causal Credit Assignment                    │      |
|  └─────────────────────────────────────────────────────────┘      |
|           ↓ Policy Updates                                        |
|  Module 4: HYPERGRAPH-FEDERATED AGGREGATION (HFA)                |
|  ┌─────────────────────────────────────────────────────────┐      |
|  │  Causal Contribution Weighting → Hyperedge-Aware        │      |
|  │  Aggregation → Non-IID Correction via Causal Transport  │      |
|  └─────────────────────────────────────────────────────────┘      |
|           ↓ Global Model Update                                   |
|  Module 5: DIGITAL TWIN CAUSAL SIMULATOR (DTCS)                  |
|  ┌─────────────────────────────────────────────────────────┐      |
|  │  Environment Cloning → Interventional Simulation        │      |
|  │  → Counterfactual Trajectory Generation                 │      |
|  └─────────────────────────────────────────────────────────┘      |
+===================================================================+
```

### 4.3 Module 1: Causal Hypergraph State Encoder (CHSE)

**Step 1 — Dynamic Hypergraph Construction.** At time $t$, construct $\mathcal{H}(t)$ from observed system telemetry. Vertices are typed: $v \in \mathcal{V}$ has type $\text{tp}(v) \in \{\texttt{device}, \texttt{edge}, \texttt{task}, \texttt{channel}\}$. Hyperedges are formed by identifying groups of entities participating in a common causal mechanism (e.g., a task-device-channel-node offloading tuple).

**Step 2 — Online Causal Discovery on Hypergraphs.** We extend the PC algorithm to hypergraphs. Define conditional independence on hyperedges:

$$e_1 \perp\!\!\!\perp_{\mathcal{H}} e_2 \mid \mathbf{S} \iff P(e_1, e_2 \mid \mathbf{S}) = P(e_1 \mid \mathbf{S}) P(e_2 \mid \mathbf{S})$$

where $\mathbf{S} \subset \mathcal{E}(t)$ is a separating set of hyperedges. We use a kernel-based conditional independence test (KCIT) adapted for spike-coded signals, updated online via a sliding window of size $W$.

**Step 3 — Hypergraph Neural Attention Encoding.** We define a novel **Causal Hypergraph Attention Network (CHAN):**

$$\mathbf{h}_v^{(\ell+1)} = \sigma\left( \sum_{e \in \mathcal{E}(v)} \alpha_{v,e}^{(\ell)} \cdot \sum_{u \in e \setminus \{v\}} \mathbf{W}_{\text{tp}(u)}^{(\ell)} \mathbf{h}_u^{(\ell)} \cdot \psi(\mathcal{G}, v, u) \right)$$

where:
- $\mathcal{E}(v) = \{e \in \mathcal{E} : v \in e\}$ are hyperedges incident to $v$.
- $\alpha_{v,e}^{(\ell)}$ is an attention coefficient computed via:

$$\alpha_{v,e}^{(\ell)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v^{(\ell)} \| \mathbf{W}\bar{\mathbf{h}}_e^{(\ell)} \| \mathbf{c}_{v,e}]\right)\right)}{\sum_{e' \in \mathcal{E}(v)} \exp(\cdots)}$$

- $\mathbf{c}_{v,e}$ is a **causal encoding vector** derived from the SCM: $\mathbf{c}_{v,e} = \text{Enc}(\text{do}(v), \text{effect}(e \setminus \{v\}))$, encoding the interventional effect of $v$ on the other members of hyperedge $e$.
- $\psi(\mathcal{G}, v, u)$ is a **causal gate** that modulates message passing by the causal strength from $v$ to $u$:

$$\psi(\mathcal{G}, v, u) = \sigma\left(\mathbf{w}_\psi^T \cdot [\text{ACE}(v \rightarrow u) \| \text{NDE}(v \rightarrow u)]\right)$$

where $\text{ACE}$ is the average causal effect and $\text{NDE}$ is the natural direct effect, both estimated from the SCM.

The final state embedding is: $\mathbf{z}(t) = \text{ReadOut}(\{\mathbf{h}_v^{(L)}\}_{v \in \mathcal{V}(t)})$ using a Set2Set pooling mechanism.

### 4.4 Module 2: Spiking Policy Network (SPN)

Each agent $i$ maintains a spiking neural network policy $\pi_i^{\text{SNN}}$ parameterized by $\boldsymbol{\omega}_i$.

**Spike Encoding Layer.** The continuous state embedding $\mathbf{z}_i(t)$ is converted to spike trains via a **learned rate-temporal hybrid coding**:

$$s_j(t') = \begin{cases} 1 & \text{if } \int_{t'-\Delta}^{t'} \left(\beta_j z_{i,j}(t) + (1-\beta_j) \frac{dz_{i,j}}{dt}\right) dt' \geq \vartheta_j \\ 0 & \text{otherwise} \end{cases}$$

where $\beta_j \in [0,1]$ is a learnable parameter balancing rate coding ($\beta_j = 1$) and temporal coding ($\beta_j = 0$), and $\vartheta_j$ is an adaptive threshold.

**Spiking Hidden Layers.** We use Leaky Integrate-and-Fire (LIF) neurons with learnable parameters:

$$u_j^{(\ell)}(t'+1) = \lambda_j u_j^{(\ell)}(t') + \sum_{k} w_{jk}^{(\ell)} s_k^{(\ell-1)}(t') - \vartheta_j s_j^{(\ell)}(t')$$

$$s_j^{(\ell)}(t') = \Theta(u_j^{(\ell)}(t') - \vartheta_j)$$

where $\lambda_j$ is the membrane time constant, $\Theta$ is the Heaviside function (with surrogate gradient $\tilde{\Theta}'(x) = \frac{1}{\pi} \frac{1}{1+(x\pi)^2}$ for backpropagation).

**Temporal Difference Spike Timing (TDST) Decoder.** Actions are decoded from output spike trains via a novel mechanism that leverages spike timing differences:

$$a_i^{(m)}(t) = \text{softmax}\left(\sum_{t'=1}^{T_s} \kappa(t') \cdot \mathbf{s}^{(L)}(t')\right)_m, \quad \kappa(t') = \exp(-\beta_{\kappa}(T_s - t'))$$

where later spikes carry more weight (recency bias), mimicking neurobiological decision-making. For continuous actions (power control, resource allocation), we use population coding over the output neurons.

**Energy Model.** The computational energy of the SNN policy is:

$$E_i^{\text{SNN}}(t) = E_{\text{AC}} \cdot \underbrace{\sum_{\ell, j, t'} s_j^{(\ell)}(t')}_{\text{total spikes}} \ll E_{\text{MAC}} \cdot \underbrace{\sum_{\ell} n_\ell \cdot n_{\ell+1} \cdot T_s}_{\text{ANN equivalent}}$$

where $E_{\text{AC}} \approx 0.9$ pJ and $E_{\text{MAC}} \approx 4.6$ pJ on 45nm technology, giving a theoretical $5\times$ or greater energy reduction at typical spike sparsity ($< 10\%$ firing rate).

### 4.5 Module 3: Causal Counterfactual Policy Gradient (CCPG)

This is the core learning algorithm. Standard MARL suffers from credit assignment issues (each agent's reward is confounded by others' actions). We address this causally.

**Interventional Value Function.** For agent $i$ under objective $m$:

$$Q_i^{(m)}(\mathbf{s}, \text{do}(a_i), \mathcal{H}) = \mathbb{E}\left[\sum_{t'=t}^{\infty} \gamma^{t'-t} R_i^{(m)}(t') \mid \mathbf{s}(t) = \mathbf{s}, \text{do}(A_i(t) = a_i), \mathcal{H}(t)\right]$$

The $\text{do}(\cdot)$ operator (Pearl's intervention) means we evaluate the effect of *setting* agent $i$'s action to $a_i$, severing incoming causal edges to $A_i$ in the SCM — thereby removing confounding from correlated agent behaviors.

**Counterfactual Baseline.** To reduce variance, we define:

$$b_i^{(m)}(\mathbf{s}, \mathcal{H}) = \mathbb{E}_{a_i \sim \pi_i^{\text{cf}}} \left[Q_i^{(m)}(\mathbf{s}, \text{do}(a_i), \mathcal{H})\right]$$

where $\pi_i^{\text{cf}}$ is a counterfactual policy obtained by marginalizing over agent $i$'s action while keeping others fixed (COMA-style, but with causal interventions rather than conditional expectations).

**Causal Advantage:**

$$A_i^{(m), \text{causal}}(\mathbf{s}, a_i, \mathcal{H}) = Q_i^{(m)}(\mathbf{s}, \text{do}(a_i), \mathcal{H}) - b_i^{(m)}(\mathbf{s}, \mathcal{H})$$

**Multi-Objective Scalarization.** We use a **Causal Tchebycheff scalarization** that adapts weights based on causal importance:

$$A_i^{\text{scal}}(\mathbf{s}, a_i, \mathcal{H}) = \max_{m=1,\dots,4} \left\{ \lambda_m(t) \cdot \frac{A_i^{(m), \text{causal}}(\mathbf{s}, a_i, \mathcal{H})}{\sigma_m} \right\}$$

where $\lambda_m(t)$ are time-varying preference weights adapted via:

$$\lambda_m(t+1) = \lambda_m(t) + \eta_\lambda \cdot \left(\text{ACE}(\text{objective}_m \rightarrow \text{system\_utility}) - \bar{\lambda}_m\right)$$

This causally adjusts the objective weights by how much each objective *causally affects* overall system utility, not merely correlates with it.

**Policy Gradient Update.** For agent $i$ with spiking policy $\pi_i^{\text{SNN}}(\boldsymbol{\omega}_i)$:

$$\nabla_{\boldsymbol{\omega}_i} J_i = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t} \nabla_{\boldsymbol{\omega}_i} \log \pi_i^{\text{SNN}}(a_i(t) | \mathbf{z}_i(t); \boldsymbol{\omega}_i) \cdot A_i^{\text{scal}}(\mathbf{s}(t), a_i(t), \mathcal{H}(t))\right]$$

The gradient through the SNN uses the surrogate gradient $\tilde{\Theta}'$ applied via backpropagation through time (BPTT) over the spike time steps.

**Lagrangian Constraint Handling.** For constraints C1–C6, we augment the objective:

$$\mathcal{L}(\boldsymbol{\omega}, \boldsymbol{\mu}) = J(\boldsymbol{\omega}) + \sum_{j=1}^{6} \mu_j \cdot g_j(\boldsymbol{\omega})$$

where $g_j$ are constraint violation functions and $\mu_j \geq 0$ are dual variables updated via:

$$\mu_j \leftarrow \left[\mu_j + \eta_\mu \cdot g_j(\boldsymbol{\omega})\right]^+$$

### 4.6 Module 4: Hypergraph-Federated Aggregation (HFA)

Standard FedAvg aggregates as $\mathbf{w}^{(r+1)} = \sum_i \frac{|D_i|}{|D|} \mathbf{w}_i^{(r)}$. This ignores (a) non-IID data heterogeneity and (b) the causal structure of the system.

**Causal Contribution Weighting.** We define:

$$\alpha_i^{\text{causal}}(r) = \frac{\text{ACE}(\mathbf{w}_i^{(r)} \rightarrow F(\mathbf{w}^{(r+1)}))}{\sum_{j} \text{ACE}(\mathbf{w}_j^{(r)} \rightarrow F(\mathbf{w}^{(r+1)}))}}$$

The ACE is estimated by interventional experiments in the Digital Twin (Module 5): simulating what happens to the global loss when we intervene on each client's model contribution.

**Hyperedge-Aware Aggregation.** Clients that share hyperedges (i.e., are causally linked) should have correlated model updates. We aggregate within hyperedges first, then across:

$$\bar{\mathbf{w}}_e^{(r)} = \frac{1}{|e|} \sum_{i \in e} \mathbf{w}_i^{(r)}, \quad \mathbf{w}^{(r+1)} = \sum_{e \in \mathcal{E}_{\text{FL}}} \frac{\alpha_e^{\text{causal}} \cdot |e|}{Z} \bar{\mathbf{w}}_e^{(r)}$$

where $\mathcal{E}_{\text{FL}} \subseteq \mathcal{E}$ are hyperedges relevant to the FL process.

**Non-IID Correction via Causal Optimal Transport.** To mitigate non-IID drift, we compute an optimal transport plan $\boldsymbol{\Pi}_i$ between each client's gradient distribution $\nabla F_i$ and the global gradient $\nabla F$, but constrained to the causal graph:

$$\boldsymbol{\Pi}_i^* = \arg\min_{\boldsymbol{\Pi} \in \mathcal{U}(\nabla F_i, \nabla F)} \langle \boldsymbol{\Pi}, \mathbf{C}_i^{\text{causal}} \rangle$$

where $\mathbf{C}_i^{\text{causal}}$ is a cost matrix that penalizes transport plans violating the causal structure (e.g., moving gradient mass between causally independent parameter groups).

### 4.7 Module 5: Digital Twin Causal Simulator (DTCS)

A differentiable Digital Twin $\mathfrak{D}$ mirrors the physical edge system:

$$\hat{\mathbf{s}}(t+1) = \mathfrak{D}(\mathbf{s}(t), \mathbf{A}(t), \mathcal{H}(t); \boldsymbol{\xi})$$

The twin serves three purposes:
1. **Counterfactual training data:** Generate trajectories under interventions $\text{do}(A_i = a)$ that may be too costly to execute physically.
2. **Causal discovery validation:** Test discovered causal edges by performing interventions in the twin and checking predictions.
3. **Safe exploration:** Agents explore in the twin before deploying actions to the real system, implementing a **sim-to-real causal transfer** protocol.

The twin is updated from real system observations via:

$$\boldsymbol{\xi}^{(t+1)} = \boldsymbol{\xi}^{(t)} - \eta_{\xi} \nabla_{\boldsymbol{\xi}} \|\mathbf{s}^{\text{real}}(t+1) - \mathfrak{D}(\mathbf{s}(t), \mathbf{A}(t), \mathcal{H}(t); \boldsymbol{\xi})\|^2$$

### 4.8 Complete Algorithm: Pseudocode

```
Algorithm: CHRONOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: N edge agents, initial model w⁰, hypergraph H⁰,
       spiking policy parameters {ωᵢ⁰}, dual variables μ⁰,
       Digital Twin D, learning rates η_ω, η_μ, η_ξ, η_λ

Output: Trained policies {πᵢ*}, optimized global model w*

 1:  Initialize CHSE, SPN, CCPG, HFA, DTCS modules
 2:  for round r = 1, 2, ..., R do
 3:  │
 4:  │  ▶ PHASE 1: CAUSAL HYPERGRAPH UPDATE
 5:  │  Collect system telemetry from all nodes
 6:  │  Construct H(t) with heterogeneous vertex/hyperedge types
 7:  │  Run online causal discovery (sliding window KCIT)
 8:  │  Update SCM G(t) structural equations
 9:  │
10:  │  ▶ PHASE 2: STATE ENCODING
11:  │  for each agent i ∈ N in parallel do
12:  │  │  Compute local observation oᵢ(t)
13:  │  │  Encode via CHAN: zᵢ(t) = CHSE(oᵢ(t), H(t), G(t))
14:  │  │  Spike-encode: sᵢ(t) = SpikeEncode(zᵢ(t); φᵢ)
15:  │  end for
16:  │
17:  │  ▶ PHASE 3: SPIKING POLICY EXECUTION
18:  │  for each agent i ∈ N in parallel do
19:  │  │  Forward pass through LIF layers (Tₛ spike steps)
20:  │  │  Decode action: aᵢ(t) = TDST_Decode(s_out_i)
21:  │  │  // aᵢ = (xᵢ, rᵢ, pᵢ, aᵢ_ch, θᵢ, φᵢ)
22:  │  end for
23:  │
24:  │  ▶ PHASE 4: DIGITAL TWIN SIMULATION
25:  │  Simulate joint action A(t) in Digital Twin D
26:  │  Generate K counterfactual trajectories per agent:
27:  │  │  for each agent i, sample a'ᵢ ~ πᵢ^cf
28:  │  │  τ_cf_i = D.simulate(s(t), do(Aᵢ = a'ᵢ), A₋ᵢ)
29:  │  Validate causal structure: check G(t) predictions vs D
30:  │
31:  │  ▶ PHASE 5: ENVIRONMENT EXECUTION
32:  │  Execute A(t) in real system (after safety check in twin)
33:  │  Observe rewards {Rᵢ^(m)}_{m=1..4}, next state s(t+1)
34:  │  Update Digital Twin parameters ξ from real observations
35:  │
36:  │  ▶ PHASE 6: CAUSAL POLICY GRADIENT UPDATE
37:  │  for each agent i ∈ N in parallel do
38:  │  │  Compute interventional Q: Qᵢ^(m)(s, do(aᵢ), H)
39:  │  │  Compute counterfactual baseline: bᵢ^(m)(s, H)
40:  │  │  Compute causal advantage: Aᵢ^causal
41:  │  │  Scalarize: Aᵢ^scal using adaptive λ(t)
42:  │  │  Update ωᵢ via surrogate-gradient BPTT:
43:  │  │  │  ωᵢ ← ωᵢ - η_ω ∇_ωᵢ L(ωᵢ, μ)
44:  │  │  Update dual: μⱼ ← [μⱼ + η_μ gⱼ(ω)]⁺
45:  │  end for
46:  │  Update λ_m via causal importance (ACE-based)
47:  │
48:  │  ▶ PHASE 7: HYPERGRAPH-FEDERATED AGGREGATION
49:  │  if r mod τ_agg == 0 then
50:  │  │  Compute causal contribution weights αᵢ^causal
51:  │  │  Aggregate within hyperedges, then across
52:  │  │  Apply non-IID correction via causal OT
53:  │  │  Broadcast updated global model w^(r+1)
54:  │  end if
55:  │
56:  end for
57:  return {πᵢ*}, w*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 5. Key Innovations

### Innovation 1: Causal Hypergraph State Representation with Interventional Message Passing

**What exists:** GNN-based representations for edge computing (e.g., scheduling on computation graphs) use pairwise edges and correlational attention. Hypergraph neural networks exist (HGNN, HyperGCN) but are not causal.

**What is new:** CHRONOS introduces **causal gates** $\psi(\mathcal{G}, v, u)$ into hypergraph message passing, modulated by interventional quantities (ACE, NDE) from an online-learned SCM. This means the model distinguishes "node $A$ and node $B$ are both busy when traffic is high" (correlation) from "overloading node $A$ *causes* congestion at node $B$" (causation). No prior work combines hypergraph neural networks with structural causal models for edge system state encoding.

**Why publishable:** This extends both the hypergraph learning and causal ML literatures with a concrete, system-motivated application. The causal gate mechanism is a general architectural contribution applicable beyond edge computing.

### Innovation 2: Spiking Multi-Agent Reinforcement Learning with TDST Decoding

**What exists:** SNNs have been applied to single-agent RL (e.g., PopSAN, SNN-based DQN). Multi-agent RL exists (QMIX, MAPPO). These are disjoint research threads.

**What is new:** CHRONOS is the first framework to deploy **spiking neural network policies in a multi-agent setting** with a communication protocol based on spike-coded messages transmitted through the causal hypergraph. The TDST decoder is novel — it exploits spike *timing* (not just rates) for action selection, enabling temporal credit assignment at the neuronal level that complements the causal credit assignment at the agent level.

**Why publishable:** The intersection of SNNs + MARL is unexplored. The energy efficiency argument is compelling for edge deployment: SNN policies consume $5{-}10\times$ less energy than equivalent ANN policies, directly contributing to Objective 3.

### Innovation 3: Causal Counterfactual Policy Gradient (CCPG) for Multi-Objective MARL

**What exists:** COMA uses a counterfactual baseline but is not causal (uses conditional expectations, not interventions). Causal RL exists (CausalRL, Causal MCTS) but is single-agent and single-objective.

**What is new:** CCPG performs credit assignment via Pearl's do-calculus in a multi-agent, multi-objective setting. The interventional advantage $A_i^{(m), \text{causal}}$ isolates each agent's *true causal contribution* to each objective, eliminating spurious correlations from confounders. The causal Tchebycheff scalarization adapts objective weights based on causal importance rather than fixed preferences.

**Why publishable:** This provides theoretically grounded multi-agent credit assignment that is both causally valid and multi-objective — a triple novelty at the intersection of three active research areas.

### Innovation 4: Hypergraph-Federated Aggregation with Causal Optimal Transport

**What exists:** FedAvg, FedProx, SCAFFOLD address non-IID data. Graph-based FL (FedGNN) uses pairwise graphs. Optimal transport for FL exists (FedOT) but is not causally constrained.

**What is new:** HFA aggregates models *within hyperedges first* (respecting the multi-way causal structure), then across. The non-IID correction uses optimal transport constrained by the causal graph, ensuring gradient corrections do not violate causal independence assumptions.

**Why publishable:** This is a new aggregation paradigm that structurally respects both the higher-order topology and causal semantics of the edge system.

### Innovation 5: Digital Twin as a Causal Interventional Engine

**What exists:** Digital twins for network optimization (e.g., NS-3 based). Sim-to-real in RL (domain randomization).

**What is new:** The DTCS is used *specifically* for causal purposes: generating interventional data $P(\mathbf{s}' | \text{do}(a_i))$ that cannot be obtained from observational data alone. It also validates causal discovery by performing controlled interventions. This establishes a principled **causal sim-to-real transfer** framework.

**Why publishable:** This reframes digital twins from simulation tools to *causal reasoning engines*, a conceptually novel contribution.

---

## 6. Theoretical Contributions

### 6.1 Convergence of CCPG under Non-Stationary Causal Graphs

**Theorem 1 (CCPG Convergence).** *Under Assumptions A1–A4 (bounded rewards, Lipschitz transition kernel, $\epsilon$-accurate causal graph estimation, bounded causal effect estimation error $\delta_{\text{ACE}}$), the CCPG algorithm converges to an $\epsilon'$-approximate Pareto-stationary point of the CC-MOMG at a rate:*

$$\|\nabla_{\boldsymbol{\omega}} \mathcal{L}(\boldsymbol{\omega}^{(T)}, \boldsymbol{\mu}^{(T)})\| \leq \mathcal{O}\left(\frac{1}{\sqrt{T}} + \delta_{\text{ACE}} + \epsilon_{\mathcal{H}}\right)$$

*where $\epsilon_{\mathcal{H}}$ is the hypergraph estimation error and $T$ is the number of iterations.*

**Proof sketch:** The proof extends the multi-objective policy gradient theorem (Parisi et al.) to interventional distributions. Key steps:
1. Show that the interventional advantage is an unbiased estimator of the true causal effect (by validity of the SCM).
2. Bound the variance reduction from the counterfactual baseline.
3. Handle non-stationarity of $\mathcal{H}(t)$ via a tracking argument bounding $\|\mathcal{H}(t) - \mathcal{H}(t-1)\|$ and its propagation through the CHAN encoder.
4. Apply the Lagrangian dual convergence results of Paternain et al. for constrained MDPs.

### 6.2 Regret Bound for Online Causal Hypergraph Discovery

**Theorem 2 (Causal Discovery Regret).** *The online causal discovery procedure incurs cumulative regret (in terms of incorrect causal edges) bounded by:*

$$\text{Regret}_{\text{causal}}(T) \leq \mathcal{O}\left(|\mathcal{E}|^2 \cdot \sqrt{T \log T}\right)$$

*relative to an oracle with perfect knowledge of the true causal graph at each time step.*

This follows from adapting online learning regret bounds (Hazan) to the causal discovery setting, treating each conditional independence test as a stochastic bandit arm.

### 6.3 Federated Convergence under Causal Aggregation

**Theorem 3 (HFA Convergence).** *Under $L$-smooth, $\mu$-strongly convex local objectives with non-IID parameter $\zeta^2 = \sum_i \frac{|D_i|}{|D|} \|\nabla F_i(\mathbf{w}^*) - \nabla F(\mathbf{w}^*)\|^2$, the HFA achieves:*

$$\mathbb{E}[F(\mathbf{w}^{(R)})] - F(\mathbf{w}^*) \leq \mathcal{O}\left(\frac{L}{\mu R} + \frac{\zeta^2}{\mu R} \cdot \Gamma_{\mathcal{H}}\right)$$

*where $\Gamma_{\mathcal{H}} = 1 - \frac{\lambda_2(\mathbf{L}_{\mathcal{H}})}{\lambda_{\max}(\mathbf{L}_{\mathcal{H}})} \in [0, 1)$ is the spectral gap of the hypergraph Laplacian $\mathbf{L}_{\mathcal{H}}$, with $\Gamma_{\mathcal{H}} = 0$ for a fully connected hypergraph (recovering FedAvg rates) and $\Gamma_{\mathcal{H}} \to 1$ for sparse hypergraphs.*

**Interpretation:** The hypergraph structure provides a tighter convergence rate than FedAvg when the hypergraph captures the true non-IID structure. The causal OT correction reduces $\zeta^2$ by aligning gradient distributions.

### 6.4 SNN Policy Approximation Guarantee

**Theorem 4 (SNN Universal Approximation).** *For any $\epsilon > 0$ and any continuous policy $\pi: \mathcal{S} \rightarrow \Delta(\mathcal{A})$, there exists an SNN policy $\pi^{\text{SNN}}$ with $\mathcal{O}(1/\epsilon^2)$ LIF neurons and $T_s = \mathcal{O}(\log(1/\epsilon))$ time steps such that:*

$$\sup_{\mathbf{s} \in \mathcal{S}} \text{KL}(\pi(\cdot | \mathbf{s}) \| \pi^{\text{SNN}}(\cdot | \mathbf{s})) \leq \epsilon$$

*with energy cost $E^{\text{SNN}} \leq \mathcal{O}(\epsilon^{-1} \cdot E_{\text{AC}} \cdot \log(1/\epsilon))$ compared to ANN energy $E^{\text{ANN}} = \mathcal{O}(\epsilon^{-2} \cdot E_{\text{MAC}})$.*

### 6.5 Discussion: Non-IID and Dynamic Environments

**Non-IID Data.** The causal OT correction in HFA explicitly addresses non-IID heterogeneity. The correction strength is modulated by the causal structure: clients that are causally independent can have arbitrarily different data distributions without affecting convergence, because their updates are aggregated through separate hyperedges. Only clients within the same causal hyperedge need aligned distributions.

**Dynamic Environments.** CHRONOS handles non-stationarity at three levels:
1. **Causal graph dynamics:** The online causal discovery module tracks changes in system topology (node joins/leaves, channel variations) with bounded tracking regret (Theorem 2).
2. **Task dynamics:** The spiking policy inherently handles event-driven inputs; the LIF membrane dynamics act as temporal filters smoothing transient fluctuations.
3. **Concept drift in FL:** The causal contribution weights $\alpha_i^{\text{causal}}$ automatically down-weight clients whose local model has diverged causally from the global objective, providing robustness to distribution shifts.

---

## 7. System Architecture

### 7.1 Physical Layer: Edge Nodes

```
┌─────────────────────────────────────────────────────────────┐
│                    EDGE NODE i                              │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐     │
│  │ IoT Iface│  │ Compute Unit │  │ Neuromorphic Core │     │
│  │ (sensors,│  │ (CPU/GPU for │  │ (SNN inference on │     │
│  │  actuator│  │  task exec,  │  │  Intel Loihi or   │     │
│  │  gateway)│  │  FL training)│  │  SpiNNaker chip)  │     │
│  └────┬─────┘  └──────┬───────┘  └────────┬──────────┘     │
│       │               │                    │                │
│  ┌────┴───────────────┴────────────────────┴──────────┐     │
│  │              Local CHRONOS Agent                    │     │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────────────┐   │     │
│  │  │ CHSE    │ │ SPN      │ │ Local FL Trainer   │   │     │
│  │  │ (local) │ │ (policy) │ │ (SGD on local D_i) │   │     │
│  │  └─────────┘ └──────────┘ └───────────────────┘   │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

Each edge node maintains:
- A local copy of the CHSE encoder (processes its neighborhood of the hypergraph).
- A spiking policy network (on neuromorphic hardware if available, else emulated).
- A local FL training process with standard GPU/CPU.
- Telemetry buffers for system state, channel measurements, task queues.

### 7.2 Communication Layer

```
┌─────────────────────────────────────────────────────────────┐
│                  COMMUNICATION LAYER                        │
│                                                             │
│  ┌──────────────────────┐    ┌────────────────────────┐    │
│  │  Spike-Coded Message │    │  Gradient Compression  │    │
│  │  Protocol (SCMP)     │    │  & Sparsification      │    │
│  │  - Binary spike      │    │  - Top-K + quantize    │    │
│  │    trains as control  │    │  - Causal mask: only   │    │
│  │    signals            │    │    transmit causally   │    │
│  │  - Ultra-low overhead │    │    relevant params     │    │
│  │    (~bits, not floats)│    │  - Hyperedge routing   │    │
│  └──────────────────────┘    └────────────────────────┘    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Channel-Aware Scheduling                            │  │
│  │  - OFDMA resource blocks allocated per hyperedge     │  │
│  │  - Power control from SPN actions                    │  │
│  │  - Interference managed via causal graph (C6)        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Key protocol innovation — **Spike-Coded Message Protocol (SCMP):** Inter-agent coordination messages are encoded as binary spike trains rather than floating-point vectors. A message $\mathbf{m}_{i \to j}$ is a binary sequence of length $T_s$ on each of $d$ channels, requiring $d \cdot T_s$ bits instead of $d \cdot 32$ bits for float32 vectors. At typical $T_s = 16$ and $d = 64$, this is $64 \times 16 = 1024$ bits = 128 bytes vs. $64 \times 32 = 2048$ bits = 256 bytes — a $2\times$ reduction, with the actual reduction being higher due to spike sparsity (typically $<10\%$ of bits are 1).

### 7.3 Learning Layer

The learning layer consists of three parallel processes:

1. **Federated Model Training.** Standard distributed SGD with HFA aggregation. Each node trains on its local data for $\kappa_i$ steps before uploading compressed gradients.

2. **Policy Learning (S-MARL).** The CCPG algorithm trains spiking policies. Executed in the Digital Twin for exploration, with periodic deployment to the real system.

3. **Causal Structure Learning.** Continuous online causal discovery updating the SCM. This is computationally lightweight (kernel-based conditional independence tests on summary statistics) and runs as a background process.

### 7.4 Decision Layer

The decision layer integrates outputs from all modules:

```
           ┌─────────────────────────────────────┐
           │         DECISION LAYER              │
           │                                     │
           │  Input: z(t), H(t), G(t)           │
           │         SPN output spikes           │
           │         Constraint status           │
           │                                     │
           │  ┌──────────────┐                   │
           │  │ Action Parser │ → x(t): offload  │
           │  │              │ → r(t): resource   │
           │  │ Spike → Real │ → p(t): power      │
           │  │              │ → a(t): channel     │
           │  │              │ → θ(t): FL params   │
           │  │              │ → φ(t): SNN params  │
           │  └──────┬───────┘                    │
           │         │                            │
           │  ┌──────▼───────┐                    │
           │  │ Causal Safety│                    │
           │  │   Filter     │                    │
           │  │ Enforce C6:  │                    │
           │  │ A(t) ∈ A^csl │                    │
           │  └──────┬───────┘                    │
           │         │                            │
           │  ┌──────▼───────┐                    │
           │  │   Executor   │ → Physical system  │
           │  └──────────────┘                    │
           └─────────────────────────────────────┘
```

The **Causal Safety Filter** is a critical component: before any action is executed, it checks consistency with the learned causal graph. If the action would violate a causal invariance (e.g., offloading to a node whose causal parents indicate imminent failure), it projects the action to the nearest causally valid action:

$$\mathbf{A}^{\text{safe}}(t) = \arg\min_{\mathbf{A}' \in \mathcal{A}^{\text{causal}}(\mathcal{H}(t))} \|\mathbf{A}'- \mathbf{A}(t)\|^2$$

---

## 8. Experimental Plan

### 8.1 Simulation Environment

**Primary Platform: CHRONOS-Sim** (to be developed), a Digital Twin simulation environment integrating:
- **Network simulation:** NS-3 for wireless channel modeling (Rayleigh/Rician fading, path loss, OFDMA).
- **Compute simulation:** Containerized edge nodes with configurable CPU/GPU/memory, realistic task execution times.
- **FL simulation:** Built on Flower framework with customizable non-IID data partitioning (Dirichlet distribution, label skew, quantity skew).
- **SNN simulation:** Norse (PyTorch-based SNN library) or snnTorch for spiking neural network policy emulation.
- **Causal engine:** DoWhy + custom hypergraph causal discovery.

**Scenarios:**
1. **Smart Factory:** 50 edge nodes, 200 IoT sensors, object detection + anomaly detection tasks, industrial wireless (5G URLLC).
2. **Autonomous Vehicle Fleet:** 20 vehicles + 10 RSUs, federated perception model, V2X communication, high mobility.
3. **Smart Hospital:** 30 edge nodes, 500 wearable IoT devices, patient monitoring + diagnostic ML, WiFi-6.

### 8.2 Datasets

| Scenario | Dataset | Task |
|---|---|---|
| Smart Factory | MVTec-AD, CIFAR-100 | Anomaly detection, classification |
| Vehicle Fleet | nuScenes, BDD100K | 3D object detection, segmentation |
| Hospital | MIMIC-III, PTB-XL | Time-series prediction, ECG classification |

Non-IID partitioning via Dirichlet distribution $\text{Dir}(\alpha)$ with $\alpha \in \{0.1, 0.5, 1.0, 10.0\}$ controlling heterogeneity severity.

### 8.3 Baselines

| Category | Method | Reference |
|---|---|---|
| FL Only | FedAvg, FedProx, SCAFFOLD, FedNova | McMahan 2017, Li 2020, Karimireddy 2020 |
| Task Offloading Only | DRL-Offload, HEFT, CPOP | Mao 2017, Topcuoglu 2002 |
| Joint FL + Offload | JAFL, FedEdge | Yang 2021, Wang 2022 |
| MARL | QMIX, MAPPO, MADDPG | Rashid 2018, Yu 2022, Lowe 2017 |
| GNN-based | GNN-Sched, HetGNN-Edge | Wang 2023, Zhang 2023 |
| SNN-based | PopSAN, SNN-DQN | Tang 2021, Chen 2022 |
| Causal RL | Causal MBRL, CCM | Lu 2018, Pitis 2020 |

**Ablation variants of CHRONOS:**
- CHRONOS-noSNN (replace SNN with MLP policies)
- CHRONOS-noHG (replace hypergraph with standard graph)
- CHRONOS-noCausal (remove causal components, use correlational)
- CHRONOS-noFed (centralized learning)
- CHRONOS-noDT (no Digital Twin, learn from real system only)

### 8.4 Metrics

| Metric | Formula/Description | Target |
|---|---|---|
| **Model Accuracy** | Test accuracy / F1-score of federated model | Maximize |
| **End-to-End Latency** | $\frac{1}{K}\sum_k \tau_k^{\text{total}}$, plus tail latency (p99) | Minimize |
| **Energy Consumption** | $\sum_i E_i^{\text{total}}$ (Joules) | Minimize |
| **Communication Cost** | Total bits transmitted per round | Minimize |
| **Deadline Violation Rate** | $\frac{1}{K}\sum_k \mathbb{1}[\tau_k > \tau_k^{\max}]$ | Minimize |
| **Pareto Optimality** | Hypervolume indicator of multi-objective front | Maximize |
| **Convergence Speed** | Rounds to reach target accuracy | Minimize |
| **Fairness** | Jain's fairness index across agents | Maximize |
| **SNN Spike Sparsity** | Average firing rate across agents | Monitor |
| **Causal Discovery F1** | Precision/recall of recovered causal edges | Monitor |

### 8.5 Evaluation Protocol

1. **Convergence analysis:** Plot all objectives vs. training rounds. Compare convergence speed and final performance against baselines.
2. **Pareto front visualization:** 2D and 3D projections of the multi-objective Pareto front. Compare hypervolume.
3. **Scalability:** Vary $N \in \{10, 20, 50, 100, 200\}$ and measure wall-clock time, communication, and performance.
4. **Non-IID robustness:** Vary $\alpha \in \{0.1, 0.5, 1.0, 10.0\}$ and measure accuracy degradation.
5. **Dynamic scenario:** Introduce node churn (random joins/leaves), channel changes, and task bursts. Measure adaptation speed.
6. **Ablation study:** Systematically disable each module to quantify its contribution.
7. **Energy profiling:** Deploy SNN policies on Intel Loihi 2 (if hardware available) or use neuromorphic energy model to estimate real-world savings.

---

## 9. Expected Results

### 9.1 Quantitative Expectations

Based on the theoretical analysis and the identified gaps in existing methods, we expect:

| Metric | vs. FedAvg+DRL | vs. MARL (MAPPO) | vs. GNN-Sched | Reason |
|---|---|---|---|---|
| Accuracy | +3–5% | +2–4% | +4–7% | Causal aggregation reduces non-IID degradation |
| Latency | -20–30% | -15–25% | -10–20% | Joint optimization vs. sequential |
| Energy | -40–60% | -35–50% | -30–45% | SNN policies + causal pruning of unnecessary communication |
| Comm. Cost | -50–70% | -40–55% | -25–40% | Spike coding + causal masking of irrelevant gradients |
| Deadline Violations | -60–80% | -40–60% | -30–50% | Causal awareness of congestion propagation |

### 9.2 Qualitative Expectations

1. **Causal hypergraph representation will outperform correlational graphs** because edge systems exhibit confounding (e.g., correlated load patterns due to shared external factors). The causal approach will correctly identify actionable relationships, avoiding spurious correlations that mislead baselines.

2. **SNN policies will show comparable decision quality to ANN policies** at a fraction of the energy cost. The accuracy-energy tradeoff curve will show an "elbow" where SNN policies achieve $>95\%$ of ANN accuracy at $<20\%$ of the energy cost.

3. **The ablation study will show each module contributes significantly**, with the causal components (CHSE causal gates, CCPG, causal OT) providing the largest gains in non-IID and dynamic scenarios, and the SNN components providing the largest gains in energy.

4. **CHRONOS will scale sub-linearly** in communication cost with the number of agents (due to spike coding and causal masking), while baselines scale linearly or worse.

5. **The Digital Twin will accelerate convergence** by $2{-}3\times$ compared to learning from real interactions only, by providing safe counterfactual exploration data.

---

## 10. Timeline (4-Year PhD Plan)

### Year 1: Foundations and Module Development

| Quarter | Milestone |
|---|---|
| **Q1** | Literature review: causal inference, hypergraph learning, SNN-RL, federated edge systems. Problem formulation refinement. |
| **Q2** | Develop CHSE module: dynamic hypergraph construction, causal discovery integration, CHAN architecture. Prove Theorem 2 (causal discovery regret). |
| **Q3** | Develop SPN module: spike encoding/decoding, LIF policy, TDST decoder. Prove Theorem 4 (SNN approximation). |
| **Q4** | Implement CHRONOS-Sim (Phase 1): NS-3 + Flower + Norse integration. Initial experiments on CHSE and SPN independently. **Publication 1 submitted.** |

### Year 2: Core Algorithm and Theoretical Analysis

| Quarter | Milestone |
|---|---|
| **Q1** | Develop CCPG module: interventional Q-function, counterfactual baseline, causal advantage. |
| **Q2** | Prove Theorem 1 (CCPG convergence). Develop causal Tchebycheff scalarization with adaptive weights. |
| **Q3** | Develop HFA module: causal contribution weighting, hyperedge-aware aggregation, causal OT correction. Prove Theorem 3 (HFA convergence). |
| **Q4** | Integrate all modules into full CHRONOS. Initial end-to-end experiments on Smart Factory scenario. **Publication 2 submitted.** |

### Year 3: Experiments, Digital Twin, and Refinement

| Quarter | Milestone |
|---|---|
| **Q1** | Develop DTCS module (Digital Twin). Implement sim-to-real causal transfer. |
| **Q2** | Full experimental evaluation: all three scenarios, all baselines, complete ablation study. |
| **Q3** | Scalability and robustness experiments. Non-IID and dynamic environment stress tests. **Publication 3 submitted.** |
| **Q4** | Neuromorphic hardware validation (Loihi 2 / SpiNNaker). Real-world pilot experiment if possible. **Publication 4 submitted.** |

### Year 4: Extensions, Writing, Defense

| Quarter | Milestone |
|---|---|
| **Q1** | Theoretical extensions: tighter bounds, privacy analysis (differential privacy + causal guarantees). **Publication 5 submitted.** |
| **Q2** | Framework generalization: apply CHRONOS to other domains (smart grid, drone swarms). **Publication 6 submitted.** |
| **Q3** | Dissertation writing. |
| **Q4** | Defense preparation and PhD defense. |

---

## 11. Potential Publications

### Paper 1: Causal Hypergraph Attention Networks for Edge System State Representation
**Target:** NeurIPS (main conference) or AAAI
**Core contribution:** The CHSE module with causal gates, interventional message passing, and online causal discovery on dynamic hypergraphs. Standalone contribution to the graph learning and causal ML communities.
**Key results:** Show that causal hypergraph representations outperform GNN, HGNN, and correlational hypergraph baselines on edge system prediction tasks.

### Paper 2: Spiking Multi-Agent Reinforcement Learning with Causal Credit Assignment
**Target:** ICML or ICLR
**Core contribution:** The full S-MARL + CCPG framework. Novel spiking policies for multi-agent systems with causal counterfactual policy gradients. Convergence guarantees (Theorem 1).
**Key results:** Demonstrate that CCPG outperforms QMIX, MAPPO, COMA on cooperative edge optimization benchmarks, with SNN policies achieving comparable performance at 5-10x lower energy.

### Paper 3: CHRONOS: Joint Federated Learning, Task Scheduling, and Communication Optimization via Causal Hypergraph Intelligence
**Target:** INFOCOM or MobiCom
**Core contribution:** The full integrated CHRONOS system. Joint optimization formulation (CC-MOMG) and end-to-end results across all three scenarios.
**Key results:** CHRONOS outperforms all baselines on the multi-objective Pareto front across accuracy, latency, energy, and communication cost.

### Paper 4: Hypergraph-Federated Aggregation with Causal Optimal Transport for Non-IID Edge Learning
**Target:** AISTATS or IEEE JSAC
**Core contribution:** The HFA module. Causal contribution weighting, hyperedge-aware aggregation, causal OT correction. Convergence analysis (Theorem 3).
**Key results:** HFA achieves faster convergence and higher accuracy than FedAvg/FedProx/SCAFFOLD under extreme non-IID conditions, with the improvement scaling with the informativeness of the causal hypergraph structure.

### Paper 5: Causal Digital Twins for Safe Multi-Agent Exploration in Edge Networks
**Target:** KDD or NeurIPS (Datasets & Benchmarks track)
**Core contribution:** The DTCS module and CHRONOS-Sim platform. Causal sim-to-real transfer methodology. Release the simulator as an open-source benchmark.
**Key results:** Digital twin-augmented training converges 2-3x faster with 80% fewer constraint violations than real-system-only training.

### Paper 6: On the Theoretical Foundations of Causal Multi-Objective Markov Games
**Target:** JMLR or Theoretical Computer Science
**Core contribution:** Formal theory paper. Define the CC-MOMG game class. Prove convergence, regret bounds, and Pareto optimality guarantees. Extend to partial observability and non-stationary environments.
**Key results:** Establish that causal interventional reasoning provides provably tighter credit assignment and lower regret than associational baselines in multi-agent multi-objective settings.

---

## Appendix A: Notation Summary

| Symbol | Description |
|---|---|
| $\mathcal{N}, \mathcal{D}, \mathcal{T}, \mathcal{C}$ | Sets of edge nodes, IoT devices, tasks, channels |
| $\mathcal{H}(t) = (\mathcal{V}, \mathcal{E}, \mathcal{W}, \mathcal{G})$ | Dynamic causal hypergraph |
| $\mathcal{G} = (\mathbf{U}, \mathbf{V}_{\mathcal{H}}, \mathcal{F}, P(\mathbf{U}))$ | Structural causal model |
| $\mathbf{x}, \mathbf{r}, \mathbf{p}, \mathbf{a}, \boldsymbol{\theta}, \boldsymbol{\phi}$ | Decision variables |
| $\mathbf{A}(t)$ | Composite action vector |
| $\pi_i^{\text{SNN}}(\boldsymbol{\omega}_i)$ | Spiking policy of agent $i$ |
| $Q_i^{(m)}(\mathbf{s}, \text{do}(a_i), \mathcal{H})$ | Interventional Q-function |
| $A_i^{(m), \text{causal}}$ | Causal advantage |
| $\alpha_i^{\text{causal}}$ | Causal FL aggregation weight |
| $\Gamma_{\mathcal{H}}$ | Hypergraph spectral gap |
| $\mathfrak{D}$ | Digital Twin simulator |
| $\text{ACE}, \text{NDE}$ | Average causal effect, natural direct effect |
| $\text{do}(\cdot)$ | Pearl's intervention operator |

---

*This proposal presents CHRONOS as a unified, theoretically grounded, and practically motivated framework addressing a critical gap at the intersection of causal reasoning, neuromorphic computing, and distributed edge intelligence. The novelty lies not in any single component but in their principled integration through the causal hypergraph abstraction — a mathematical object rich enough to represent the multi-way, interventional relationships governing edge-IoT ecosystems.*
