# ISQL: A Unified Framework of Policy Constraint and In-Sample Offline RL

ISQL is an offline reinforcement learning framework that bridges two distinct categories of methods: policy constraint-based approaches and in-sample Q-learning.

### Usage
This code is built upon the [TD3_BC](https://github.com/sfujim/TD3_BC/) codebase. Four ISQL variants are implemented:

1. ISQL with chi-squared divergence (ISQL-chi)
2. ISQL with KL divergence (ISQL-KL)
3. ISQL with Clipped ratio (IQL)
4. ISQL with Sigmoid weights (ISQL-Sigmoid)

The paper results can be reproduced by running:
```
python isql_main.py
```
