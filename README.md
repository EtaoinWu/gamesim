# No Regret Learning Simulator

This repo simulates several online learning algorithms on multilinear games.

## Learning Algorithms

This repo implements the following learning algorithms:

- Multiplicative Weights Update (MWU)
- Gradient Descent (GD)
- Optimistic Gradient Descent (OG)
- Extragradient Method (EG)
- Optimistic FTRL with arbitrary regularizer functions (OFTRL)
  - optimized for L2 regularizer

For two-players, each one of the algorithms above can be used in an alternating fashion.

We also implement the Blum-Mansour transformation ([paper](https://www.jmlr.org/papers/volume8/blum07a/blum07a.pdf)), which converts a no-regret learning rule into a no-swap-regret learning rule.

The algorithm in [arXiv:2204.11417](https://arxiv.org/abs/2204.11417) is implemented, and the experiment results in the paper is reproduced.

## Some the results:

**Positive results on OG, EG and AltGD**:

I have been messing around with different parameters, and never found a case with a $\Omega(1)$ external regret for either OG, EG or AltGD (GD but two players alternate), under fixed learning rates. *All cases that I've experimented with **fixed learning rates** have finite external regret*. (This does not hold for normal GD.)  

Under learning rate $\eta ∝ 1/T^k$ or $\eta ∝ {(1-\varepsilon)}^T$, the external regrets are unbounded.

**Reproducing the Paper**:

The algorithms `OFTRL-LogBar` and `BM-OFTRL-LogBar` in the paper [arXiv:2204.11417] are implemented. The former has O(log T) regret, and the latter O(log T) swap regret.  
Surprisingly, it seems that the former also has logarithmic swap regret. Sadly I can't prove that.