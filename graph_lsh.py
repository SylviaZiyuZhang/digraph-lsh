import networkx as nx
import mmh3

hash_mod = 1000000


def binarize_adj_matrix(g):
    return nx.adjacency_matrix(g["graph"]).todense() > 0


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
        min_hash = float("inf")
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
        min_hash = float("inf")
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

            best_in_hash_val = float("inf")
            second_best_in_hash_val = float("inf")
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
