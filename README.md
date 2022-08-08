# No Regret Learning Simulator

This repo simulates several online learning algorithms on games.
The algorithm in [arXiv:2204.11417](https://arxiv.org/abs/2204.11417) is implemented, and the experiment results in the paper is reproduced.

## Some the results:

Under general-sum games with <=4 players, each with <=3 actions, gameplays are simulated with different learning rules (EG, OG, vanilla GD, and the two algorithms in the paper).

The utilities are i.i.d. sampled from a uniform distribution. OG and EG both uses orthogonal projection to the probability simplex.

**Negative results on OG and EG:**

OG and EG both have linear swap regret.  
The paper [arXiv:2204.11417v1] claims "$O(\log T)$ second-order path lengths" with their algorithm. OG and EG both have linear path lengths, implying that they don't converge.

**Positive results on OG and EG**:

I have been messing around with different parameters, and never found a case with a $\Omega(1)$ external regret for either OG or EG, under fixed learning rates. *All cases that I've experimented with **fixed learning rates** have finite external regret*. (This does not hold for normal GD.)  
Sometimes the swap regret grows linearly, yet the external regret grows negative-linearly in $T$.

Under learning rate $\eta ∝ 1/T^(k)$ or $\eta ∝ α^T$, the external regrets are unbounded.

**Reproducing the Paper**:

The algorithms `OFTRL-LogBar` and `BM-OFTRL-LogBar` in the paper [arXiv:2204.11417] are implemented. The former has O(log T) regret, and the latter O(log T) swap regret.  
Surprisingly, it seems that the former also has logarithmic swap regret. Sadly I can't prove that.