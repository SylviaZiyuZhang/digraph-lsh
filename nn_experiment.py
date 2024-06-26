# %%
def setup_notebook():
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.magic("load_ext autoreload")
        ipython.magic("autoreload 2")

    except:
        pass


setup_notebook()

import glob
from graph_lsh import brute_force_distance, smart_jaccard_lsh, LSHIndex
import networkx as nx
import json
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


# %%
folder = "dotfiles/sae-features_lin-effects_sum-over-pos_nsamples8192_nctx64"

if not os.path.exists("graphs_contexts.pkl"):
    num_dotfiles = len(glob.glob(f"{folder}/*.dot"))

    graphs = []
    all_contexts = []

    for i in tqdm(range(num_dotfiles)):
        dotfile = f"{folder}/{i}.dot"
        context_file = f"{folder}/{i}.json"
        try:
            graph = nx.drawing.nx_pydot.read_dot(
                open(dotfile, encoding="unicode_escape")
            )
            context = json.load(open(context_file))
        except:
            graph = None
            context = None
        graphs.append(graph)
        all_contexts.append(context)

    pickle.dump((graphs, all_contexts), open("graphs_contexts.pkl", "wb"))

# %%

graphs, all_contexts = pickle.load(open("graphs_contexts.pkl", "rb"))

# %%

# Brute force
if not os.path.exists("all_distances.npy"):
    all_distances = []
    for graph_i in tqdm(graphs):
        distances = []
        for graph_j in graphs:
            if graph_i is None or graph_j is None:
                continue
            distances.append(brute_force_distance(graph_i, graph_j))
        all_distances.append(distances)

        distances_no_none = [
            distances for distances in all_distances if len(distances) > 0
        ]

    all_distances = np.array(distances_no_none)

    np.save("all_distances.npy", distances_no_none)


# %%


all_distances = np.load("all_distances.npy")

# Set diagonal to 0
all_distances[np.diag_indices(all_distances.shape[0])] = 0
# %%


plt.hist(all_distances.flatten(), bins=100, alpha=0.5, label="Brute force")
plt.yscale("log")

# %%

non_none_graphs = [graph for graph in graphs if graph is not None]
non_none_contexts = [
    context for graph, context in zip(graphs, all_contexts) if graph is not None
]


print(len(non_none_graphs))
total_num_nodes = sum([len(g.nodes()) for g in non_none_graphs])
print(total_num_nodes / len(non_none_graphs))
total_num_edges = sum([len(g.edges()) for g in non_none_graphs])
print(total_num_edges / len(non_none_graphs))

# %%

closest_pairs = []
for i in range(len(non_none_graphs)):
    for j in range(i + 1, len(non_none_graphs)):
        closest_pairs.append((i, j, all_distances[i, j]))
closest_pairs = sorted(closest_pairs, key=lambda x: -x[2])

# %%

for i, j, dist in closest_pairs[:1000]:
    contexts_1 = [
        "".join(context["context"][-10:]) for context in non_none_contexts[i].values()
    ]
    contexts_2 = [
        "".join(context["context"][-10:]) for context in non_none_contexts[j].values()
    ]

    if len(contexts_1) < 10 or len(contexts_2) < 10:
        continue

    if contexts_1[0].endswith("\n") or contexts_2[0].endswith("\n"):
        continue

    if contexts_1[0].strip().endswith("'") or contexts_2[0].strip().endswith("'"):
        continue

    print(f"Distance: {dist}")
    print(i)
    print(j)
    print(contexts_1)
    print(contexts_2)


# %%

num_lsh_hash_funcs = 128

# Jaccard LSH
if not os.path.exists(f"all_distances_jaccard_{num_lsh_hash_funcs}.npy"):
    lsh_vectors = []
    seed = 42
    for graph_i in tqdm(graphs):
        if graph_i is None:
            lsh_vectors.append(None)
            continue
        hashes = smart_jaccard_lsh(graph_i, num_lsh_hash_funcs, seed)
        lsh_vectors.append(np.array(hashes))

    all_distances_jaccard = []
    for i in tqdm(range(len(graphs))):
        distances = []
        for j in range(len(graphs)):
            if lsh_vectors[i] is None or lsh_vectors[j] is None:
                continue
            distances.append(np.mean(lsh_vectors[i] == lsh_vectors[j]))
        all_distances_jaccard.append(distances)

    np.save(
        f"all_distances_jaccard_{num_lsh_hash_funcs}.npy",
        np.array([d for d in all_distances_jaccard if len(d) > 0]),
    )

# %%

all_distances_jaccard = np.load(f"all_distances_jaccard_{num_lsh_hash_funcs}.npy")
all_distances_jaccard[np.diag_indices(all_distances_jaccard.shape[0])] = 0

plt.hist(all_distances.flatten(), bins=100, alpha=0.5, label="Brute force")
plt.hist(all_distances_jaccard.flatten(), bins=100, alpha=0.5, label="Jaccard LSH")
plt.yscale("log")

# %%


num_same = 0

for i in range(len(non_none_graphs)):
    j = np.argmax(all_distances[i])

    estimated_j = np.argmax(all_distances_jaccard[i])

    if j == estimated_j:
        num_same += 1

    contexts_1 = [
        "".join(context["context"][-10:]) for context in non_none_contexts[i].values()
    ]
    contexts_2 = [
        "".join(context["context"][-10:]) for context in non_none_contexts[j].values()
    ]

    if len(non_none_contexts[i]) < 10 or len(non_none_contexts[j]) < 10:
        continue

    print(f"Distance: {all_distances[i, j]}")
    print(i)
    print(j)
    print(contexts_1)
    print(contexts_2)

print("HERE", num_same / len(non_none_graphs))

# %%


def output_original_graphs(i, j):
    graph_i = non_none_graphs[i]
    graph_j = non_none_graphs[j]

    print(all_distances[i, j])

    i_original_index = graphs.index(graph_i)
    j_original_index = graphs.index(graph_j)

    os.makedirs(f"outputs/{i_original_index}_{j_original_index}", exist_ok=True)
    os.system(
        f"dot -Tpng {folder}/{i_original_index}.dot > outputs/{i_original_index}_{j_original_index}/i.png"
    )
    os.system(
        f"dot -Tpng {folder}/{j_original_index}.dot > outputs/{i_original_index}_{j_original_index}/j.png"
    )

    contexts_i = non_none_contexts[i]
    contexts_j = non_none_contexts[j]

    with open(
        f"outputs/{i_original_index}_{j_original_index}/i_contexts.txt", "w"
    ) as f:
        for context in contexts_i.values():
            f.write(str(context["context"][-10:]) + "\n")
    with open(
        f"outputs/{i_original_index}_{j_original_index}/j_contexts.txt", "w"
    ) as f:
        for context in contexts_j.values():
            f.write(str(context["context"][-10:]) + "\n")


output_original_graphs(110, 425)

# %%
import time
results = []

for num_tables in [2, 4, 8, 16, 32]:
    for hashes_per_table in [1, 2, 3, 4]:
        print(f"Num tables: {num_tables}, Hashes per table: {hashes_per_table}")

        lsh_index = LSHIndex(seed=42, num_tables=num_tables, hashes_per_table=hashes_per_table)

        for i, g in tqdm(enumerate(non_none_graphs)):
            lsh_index.add_graph(g)

        num_matches = 0
        num_ended_early = 0
        num_explored = 0
        bar = tqdm(enumerate(non_none_graphs))

        start = time.time()

        for i, g in bar:
            (
                closest_found_neighbor,
                closest_found_similarity,
                explored,
                ended_early,
            ) = lsh_index.query(g)
            gt_neighbor = np.argmax(all_distances[i])
            if closest_found_neighbor == non_none_graphs[gt_neighbor]:
                num_matches += 1
            if ended_early:
                num_ended_early += 1
            num_explored += explored
            bar.set_description(
                f"Matches: {num_matches / (i + 1):.2f}, Ended early: {num_ended_early / (i + 1):.2f}, Explored: {num_explored / (i + 1):.2f}"
            )

        time_taken = time.time() - start

        results.append(
            {
                "num_tables": num_tables,
                "hashes_per_table": hashes_per_table,
                "matches": num_matches / len(non_none_graphs),
                "ended_early": num_ended_early / len(non_none_graphs),
                "explored": num_explored / len(non_none_graphs),
                "time": time_taken,
            }
        )
# %%

print(num_matches / len(non_none_graphs))


# %%

pickle.dump(results, open("results.pkl", "wb"))
# %%

import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load results
results = pickle.load(open("results.pkl", "rb"))

# Extract relevant metrics
recall = np.array([result["matches"] for result in results])
qps = np.array([len(non_none_graphs) / result["time"] for result in results])

# Identify the Pareto frontier
def pareto_frontier(X, Y):
    pareto_points = []
    for i in range(len(X)):
        is_dominated = False
        for j in range(len(X)):
            if (X[j] >= X[i] and Y[j] > Y[i]) or (X[j] > X[i] and Y[j] >= Y[i]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append((X[i], Y[i]))

    pareto_points = np.array(pareto_points)
    pareto_points = pareto_points[pareto_points[:, 0].argsort()]
    return pareto_points[:, 0], pareto_points[:, 1]

pareto_recall, pareto_qps = pareto_frontier(recall, qps)

# Plot the results and the Pareto frontier
plt.figure(figsize=(10, 6))
plt.scatter(recall, qps, label="All results", alpha=0.5)
plt.plot(pareto_recall, pareto_qps, color='blue', label="Pareto frontier", linewidth=2)

# Add a horizontal line labeled "Brute force" at 1 QPS
plt.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label="Brute force")

# Labels and title
plt.xlabel("Recall")
plt.ylabel("Queries Per Second (QPS)")
# plt.title("Pareto Frontier of Recall vs QPS")
plt.legend()
# plt.show()

plt.savefig("pareto_frontier.pdf")

