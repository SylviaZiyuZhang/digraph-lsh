# %%

from src.generators.random_dag_generator import RandomDAGGenerator
import networkx as nx
import matplotlib.pyplot as plt
import mmh3
import time
import numpy as np

def get_random_graph(num_nodes=10, edge_prob=0.2):
    return RandomDAGGenerator.generate(
        num_nodes,
        edge_prob,
        edge_weight_range=[0, 1],
        edge_noise_sd_range=[0, 1],
        out_path=None,
    )

# %%

hash_mod = 1000000

def binarize_adj_matrix(g):
    return nx.adjacency_matrix(g['graph']).todense() > 0

def brute_force_distance(g1, g2):
    sets = [set(), set()]

    for g, s in zip([g1, g2], sets):
        for node in g.nodes():
            for in_neighbor in g.predecessors(node):
                s.add((node, in_neighbor))
            for out_neighbor in g.successors(node):
                s.add((node, out_neighbor))
                for in_neighbor in g.predecessors(out_neighbor):
                    if in_neighbor != node:
                        s.add((node, in_neighbor))

    if len(sets[0]) == 0 or len(sets[1]) == 0:
        return 0
    
    jaccard_sim = len(sets[0].intersection(sets[1])) / len(sets[0].union(sets[1]))
    return jaccard_sim     


def hash(val, seed):
    return mmh3.hash(str(val), seed=seed) % hash_mod

def minhash_edgelist(edge_list, num_hashes, seed):
    hash_values = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for edge in edge_list:
            hash_val = hash(edge, seed + i)
            if hash_val < min_hash:
                min_hash = hash_val
        hash_values.append(min_hash)
    return hash_values
        

def naive_jaccard_lsh(g, num_hashes, seed):
    s = set()
    for node in g.nodes():
        for in_neighbor in g.predecessors(node):
            s.add((node, in_neighbor))
        for out_neighbor in g.successors(node):
            s.add((node, out_neighbor))
            for in_neighbor in g.predecessors(out_neighbor):
                if in_neighbor != node:
                    s.add((node, in_neighbor))
    return minhash_edgelist(s, num_hashes, seed)

def combine(hash_1, hash_2):
    if hash_1 > hash_2:
        temp = hash_1
        hash_1 = hash_2
        hash_2 = temp
    return hash_1 * hash_mod + hash_2

def smart_jaccard_lsh(g, num_hashes, seed):
    minhash_values = []
    for i in range(num_hashes):
        hashed_nodes = {node: hash(node, seed + i) for node in g.nodes()}
        min_hash = float('inf')
        for node in g.nodes():
            this_hash = hashed_nodes[node]
            num_in_neighbors = 0
            for in_neighbor in g.predecessors(node):
                num_in_neighbors += 1
                hash_val = combine(this_hash, hashed_nodes[in_neighbor])
                if hash_val < min_hash:
                    min_hash = hash_val
            for out_neighbor in g.successors(node):
                hash_val = combine(this_hash, hashed_nodes[out_neighbor])
                if hash_val < min_hash:
                    min_hash = hash_val
            if num_in_neighbors < 2:
                continue
            
            best_in_hash_val = float('inf')
            second_best_in_hash_val = float('inf')
            for in_neighbor in g.predecessors(node):
                in_hash_val = hashed_nodes[in_neighbor]
                if in_hash_val < best_in_hash_val:
                    second_best_in_hash_val = best_in_hash_val
                    best_in_hash_val = in_hash_val
                elif in_hash_val < second_best_in_hash_val:
                    second_best_in_hash_val = in_hash_val
            hash_val = combine(best_in_hash_val, second_best_in_hash_val)
            if hash_val < min_hash:
                min_hash = hash_val

        minhash_values.append(min_hash)
    return minhash_values



# %%

# Accuracy test

seed = 0
num_nodes = 100
edge_prob = 0.1
all_num_hashes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
num_trials = 100


all_slow_results = []
all_fast_results = []

for trial in range(num_trials):

    slow_results = []
    fast_results = []

    for num_hashes in all_num_hashes:
        g1 = get_random_graph(edge_prob=edge_prob, num_nodes=num_nodes)
        g2 = get_random_graph(edge_prob=edge_prob, num_nodes=num_nodes)

        start = time.time()
        true_dist = brute_force_distance(g1['graph'], g2['graph'])
        time_taken = time.time() - start
        print(f"GT Dist =  {dist:.2f}, Time = {time_taken:.2f}")

        start = time.time()
        g1_hashes = naive_jaccard_lsh(g1['graph'], num_hashes, seed)
        g2_hashes = naive_jaccard_lsh(g2['graph'], num_hashes, seed)
        dist = sum([h1 == h2 for h1, h2 in zip(g1_hashes, g2_hashes)]) / num_hashes
        time_taken = time.time() - start
        print(f"Dist =  {dist:.2f}, Time = {time_taken:.2f}")
        slow_results.append(dist - true_dist)

        start = time.time()
        g1_hashes = smart_jaccard_lsh(g1['graph'], num_hashes, seed)
        g2_hashes = smart_jaccard_lsh(g2['graph'], num_hashes, seed)
        dist = sum([h1 == h2 for h1, h2 in zip(g1_hashes, g2_hashes)]) / num_hashes
        time_taken = time.time() - start
        print(f"Dist =  {dist:.2f}, Time = {time_taken:.2f}")
        fast_results.append(dist - true_dist)

    all_slow_results.append(slow_results)
    all_fast_results.append(fast_results)

all_slow_results = np.array(all_slow_results)
all_fast_results = np.array(all_fast_results)


all_fast_results = np.abs(all_fast_results)
all_slow_results = np.abs(all_slow_results)

# %%
plt.plot(all_num_hashes, np.mean(all_slow_results, axis=0), label='Naive LSH Difference to Ground Truth')
plt.fill_between(all_num_hashes, np.mean(all_slow_results, axis=0) - np.std(all_slow_results, axis=0), np.mean(all_slow_results, axis=0) + np.std(all_slow_results, axis=0), alpha=0.2)

plt.plot(all_num_hashes, np.mean(all_fast_results, axis=0), label='Smart LSH Difference to Ground Truth')
plt.fill_between(all_num_hashes, np.mean(all_fast_results, axis=0) - np.std(all_fast_results, axis=0), np.mean(all_fast_results, axis=0) + np.std(all_fast_results, axis=0), alpha=0.2)
plt.legend()
plt.xlabel('Number of hashes')
plt.xscale('log', base=2)
plt.ylabel('Error')
plt.title(f'LSH accuracy vs number of hashes, random DAG with\n{num_nodes} nodes, {edge_prob} edge prob, {num_trials} trials')

fig = plt.gcf()

fig.savefig('lsh_accuracy.png')

plt.show()

# %%

from tqdm import tqdm

# Time test

seed = 0
all_num_nodes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
edge_prob = 0.1
num_hashes = 1

slow_times = []
fast_times = []
for num_nodes in tqdm(all_num_nodes):
    g1 = get_random_graph(edge_prob=edge_prob, num_nodes=num_nodes)
    g2 = get_random_graph(edge_prob=edge_prob, num_nodes=num_nodes)
    
    start = time.time()
    g1_hashes = naive_jaccard_lsh(g1['graph'], num_hashes, seed)
    g2_hashes = naive_jaccard_lsh(g2['graph'], num_hashes, seed)
    time_taken = time.time() - start
    slow_times.append(time_taken)
    
    start = time.time()
    g1_hashes = smart_jaccard_lsh(g1['graph'], num_hashes, seed)
    g2_hashes = smart_jaccard_lsh(g2['graph'], num_hashes, seed)
    time_taken = time.time() - start
    fast_times.append(time_taken)

# %%


plt.plot(all_num_nodes, slow_times, label='Naive LSH Time')
plt.plot(all_num_nodes, fast_times, label='Smart LSH Time')
plt.legend()
plt.xlabel('Number of nodes')
plt.xscale('log', base=2)
plt.ylabel('Time (s)')
plt.yscale('log', base=2)
plt.title(f'LSH time vs number of nodes, random DAG with {edge_prob} edge prob')
fig = plt.gcf()
fig.savefig('lsh_time.png')
plt.show()