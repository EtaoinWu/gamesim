import numpy as np

def zero_sum(rng, low=-1.0, high=1.0, size=((1,),)):
    m1 = rng.uniform(low=low, high=high, size=size)
    m2 = np.array([m1, -m1])
    return m2