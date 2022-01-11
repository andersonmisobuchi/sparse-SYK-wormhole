import numpy as np
from random import sample, seed

# Return True if list x has repeated elements
def duplicates(x):
    return any(x.count(element) > 1 for element in x)

# Return True if x is a list contained in X, where X is a list of lists
def repeated_group(x, X): 
    for y in X:
        if set(y) == set(x): 
            return True
    return False

# Need to satisfy a) each group does not include repeated indices b) there are no repeated groups
# If two conditions are satisfied, put indices from L into first list of E
def regular_hypergraph_try(N, k, q):

    L = list(np.array([[i] * int(k * q) for i in range(N)]).flatten()) # List including each vertex index kq times
    E = [[], L] # Store hyperedges in E[0]
    
    for j in range(int(k * q * N)):
        if (len(L) >= q):
            h =  sample(L, q)
            if duplicates(h) == False and repeated_group(h, E[0]) == False:
                for i in h:
                    L.remove(i)
                E[0].append(h)
    return E

# Run regular_hypergraph_try until L is empty to obtain a regular hypergraph
def regular_hypergraph(N, k, q, random_seed):
   
    seed(random_seed)
    count = 0 
    E = regular_hypergraph_try(N, k, q)
    
    while E[1] != [] and count < 10000:
        E = regular_hypergraph_try(N, k, q)
        
    hypergraph = np.sort(E[0]) # organize vertex indices in increasing order
    hypergraph = tuple(map(tuple, hypergraph)) # Convert to list
    
    return hypergraph