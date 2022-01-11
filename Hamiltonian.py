import numpy as np
from scipy.special import binom
from math import factorial
from itertools import combinations
from dynamite.extras import majorana
from dynamite.operators import op_sum, op_product

def hamiltonian_kitaev(N, q, random_seed):
    
    # Set seed to generate random couplings Jijkl
    np.random.seed(random_seed)
    
    # Interaction terms in the Hamiltonian
    comb = combinations(np.arange(N), q)
    
    # Use variance with convention J=1
    hyperedges = tuple([i for i in comb])
    couplings = np.sqrt(factorial(q-1) / (N**(q-1) * 2**q) )*np.random.randn(len(hyperedges))
    
    # Create a dictionary to map a hyperedge to the random coupling
    factor = dict(zip(hyperedges, couplings))
    
    # Evaluate majoranas before building Hamiltonian
    majs = [majorana(i) for i in range(N)]

    return op_sum((op_product(majs[i] for i in idxs).scale(factor[idxs]) for idxs in hyperedges), nshow=len(hyperedges))

def hamiltonian_sparse(N, k, q, hyperedges, random_seed):
    '''
    Build q-body SYK Hamiltonian for a system of N Majoranas with sparsity
    parameter k such that Hamiltonian is a sum of kN terms.
    '''
    # Set seed to generate random couplings Jijkl
    np.random.seed(random_seed)
    
    # Use variance with convention J=1
    p = k * N / binom(N, q)
    couplings = np.sqrt(factorial(q-1) / (p * N**(q-1) * 2**q))*np.random.randn(len(hyperedges))
    
    # Create a dictionary to map a hyperedge to the random coupling
    factor = dict(zip(hyperedges, couplings))
    
    # Evaluate majoranas before building Hamiltonian
    majs = [majorana(i) for i in range(N)]

    return op_sum((op_product(majs[i] for i in idxs).scale(factor[idxs]) for idxs in hyperedges), nshow=k*N)

# Define uncoupled Hamiltonian H_L + H_R
def hamiltonian_uncoupled(N, k, q, hyperedges, random_seed):
    
    np.random.seed(random_seed)
    # Use variance with convention J=1
    p = k*N/binom(N, q)
    couplings = np.sqrt( factorial(q-1) / (p * N**(q-1) * 2**q) )*np.random.randn(len(hyperedges))
    factor = dict(zip(hyperedges, couplings))
    majs = [majorana(i) for i in range(2*N)]
    HL = op_sum((op_product(majs[i] for i in idxs).scale(factor[idxs]) for idxs in hyperedges), nshow=len(hyperedges))
    HR = op_sum((op_product(majs[i+N] for i in idxs).scale(factor[idxs]) for idxs in hyperedges), nshow=len(hyperedges))
    return op_sum([HL, HR])

# H_int Hamiltonian
def hamiltonian_int(N, k, q, mu):
    majs = [majorana(i) for i in range(2*N)]
    HI = op_sum((op_product([majs[i], majs[i+N]]) for i in range(N)), nshow=N).scale(1j*mu/2)
    return HI

def print_tex(H):
    # Clean expression for Hamiltonian output
    from IPython.display import display, Math
    simplified_str = H.get_latex().replace('*', '').replace('-+', '-').replace('+ -', '-')
    return display(Math(simplified_str))