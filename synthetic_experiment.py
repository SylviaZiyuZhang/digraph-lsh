# %%

from src.generators.random_dag_generator import RandomDAGGenerator
import matplotlib.pyplot as plt
import time
import numpy as np
from graph_lsh import brute_force_distance, naive_jaccard_lsh, smart_jaccard_lsh


def get_random_graph(num_nodes=10, edge_prob=0.2):
    return RandomDAGGenerator.generate(
        num_nodes,
        edge_prob,
        edge_weight_range=[0, 1],
        edge_noise_sd_range=[0, 1],
        out_path=None,
    )


# %%

# Accuracy test

seed = 0
num_nodes = 100
edge_prob = 0.1
all_num_hashes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
num_trials = 1000


all_slow_results = []
all_fast_results = []

for trial in range(num_trials):
    slow_results = []
    fast_results = []

    for num_hashes in all_num_hashes:
        g1 = get_random_graph(edge_prob=edge_prob, num_nodes=num_nodes)
        g2 = get_random_graph(edge_prob=edge_prob, num_nodes=num_nodes)

        start = time.time()
        gt_dist = brute_force_distance(g1["graph"], g2["graph"])
        time_taken = time.time() - start
        print(f"GT Dist =  {gt_dist:.2f}, Time = {time_taken:.2f}")

        start = time.time()
        g1_hashes = naive_jaccard_lsh(g1["graph"], num_hashes, seed)
        g2_hashes = naive_jaccard_lsh(g2["graph"], num_hashes, seed)
        dist = sum([h1 == h2 for h1, h2 in zip(g1_hashes, g2_hashes)]) / num_hashes
        time_taken = time.time() - start
        print(f"Dist =  {dist:.2f}, Time = {time_taken:.2f}")
        slow_results.append(dist - gt_dist)

        start = time.time()
        g1_hashes = smart_jaccard_lsh(g1["graph"], num_hashes, seed)
        g2_hashes = smart_jaccard_lsh(g2["graph"], num_hashes, seed)
        dist = sum([h1 == h2 for h1, h2 in zip(g1_hashes, g2_hashes)]) / num_hashes
        time_taken = time.time() - start
        print(f"Dist =  {dist:.2f}, Time = {time_taken:.2f}")
        fast_results.append(dist - gt_dist)

    all_slow_results.append(slow_results)
    all_fast_results.append(fast_results)

all_slow_results = np.array(all_slow_results)
all_fast_results = np.array(all_fast_results)


# %%

all_fast_results_abs = np.abs(all_fast_results)
all_slow_results_abs = np.abs(all_slow_results)

# %%
plt.plot(
    all_num_hashes,
    np.mean(all_slow_results, axis=0),
    label="Naive LSH Difference to Ground Truth",
)
plt.fill_between(
    all_num_hashes,
    np.mean(all_slow_results, axis=0) - 2 * np.std(all_slow_results, axis=0),
    np.mean(all_slow_results, axis=0) + 2 * np.std(all_slow_results, axis=0),
    alpha=0.2,
)

plt.plot(
    all_num_hashes,
    np.mean(all_fast_results, axis=0),
    label="Smart LSH Difference to Ground Truth",
)
plt.fill_between(
    all_num_hashes,
    np.mean(all_fast_results, axis=0) - 2 * np.std(all_fast_results, axis=0),
    np.mean(all_fast_results, axis=0) + 2 * np.std(all_fast_results, axis=0),
    alpha=0.2,
)
plt.legend()
plt.xlabel("Number of hashes")
plt.xscale("log", base=2)
plt.ylabel("Average Error")
plt.title(
    f"LSH accuracy vs number of hashes, random DAG with\n{num_nodes} nodes, {edge_prob} edge prob, {num_trials} trials"
)

fig = plt.gcf()

fig.savefig("lsh_accuracy.pdf")

plt.show()

# %%

plt.plot(
    all_num_hashes,
    np.mean(all_slow_results_abs, axis=0),
    label="Naive LSH Difference to Ground Truth",
)
plt.fill_between(
    all_num_hashes,
    np.mean(all_slow_results_abs, axis=0) - 2 * np.std(all_slow_results_abs, axis=0),
    np.mean(all_slow_results_abs, axis=0) + 2 * np.std(all_slow_results_abs, axis=0),
    alpha=0.2,
)

plt.plot(
    all_num_hashes,
    np.mean(all_fast_results_abs, axis=0),
    label="Smart LSH Difference to Ground Truth",
)
plt.fill_between(
    all_num_hashes,
    np.mean(all_fast_results_abs, axis=0) - 2 * np.std(all_fast_results_abs, axis=0),
    np.mean(all_fast_results_abs, axis=0) + 2 * np.std(all_fast_results_abs, axis=0),
    alpha=0.2,
)
plt.legend()
plt.xlabel("Number of hashes")
plt.xscale("log", base=2)
plt.ylabel("Average Absolute Error")
plt.title(
    f"LSH accuracy vs number of hashes, random DAG with\n{num_nodes} nodes, {edge_prob} edge prob, {num_trials} trials"
)

fig = plt.gcf()

fig.savefig("lsh_accuracy_abs.pdf")

plt.show()

# %%

from tqdm import tqdm

# Time test
for edge_prob in [0.1, 0.5]:
    seed = 0
    all_num_nodes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    num_hashes = 1

    slow_times = []
    fast_times = []
    for num_nodes in tqdm(all_num_nodes):
        g1 = get_random_graph(edge_prob=edge_prob, num_nodes=num_nodes)
        g2 = get_random_graph(edge_prob=edge_prob, num_nodes=num_nodes)

        start = time.time()
        g1_hashes = naive_jaccard_lsh(g1["graph"], num_hashes, seed)
        g2_hashes = naive_jaccard_lsh(g2["graph"], num_hashes, seed)
        time_taken = time.time() - start
        slow_times.append(time_taken)

        start = time.time()
        g1_hashes = smart_jaccard_lsh(g1["graph"], num_hashes, seed)
        g2_hashes = smart_jaccard_lsh(g2["graph"], num_hashes, seed)
        time_taken = time.time() - start
        fast_times.append(time_taken)

    plt.plot(all_num_nodes, slow_times, label="Naive LSH Time")
    plt.plot(all_num_nodes, fast_times, label="Smart LSH Time")
    plt.legend()
    plt.xlabel("Number of nodes")
    plt.xscale("log", base=2)
    plt.ylabel("Time (s)")
    plt.yscale("log", base=2)
    plt.title(f"LSH time vs number of nodes, random DAG with {edge_prob} edge prob")
    fig = plt.gcf()
    fig.savefig(f"lsh_time_{edge_prob}.pdf")
    plt.show()
