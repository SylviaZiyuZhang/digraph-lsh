import argparse
import yaml
import sys

sys.path.append("../..")
from src.generators.random_dag_generator import RandomDAGGenerator
from src.generators.random_dataset_generator import RandomDatasetGenerator
import networkx as nx
import os
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import signal
from functools import partial
from src.evaluation.plotter import plotter


def kill_still_running(executor, futures, sig, frame):
    print("Signal received, cleaning up...")
    for future in futures:
        future.cancel()
    executor.shutdown(wait=True)
    print("All processes terminated.")
    exit(0)


def simulate(
    args: tuple[
        pd.DataFrame,
        dict[str, object],
        dict[str, object],
        str,
        str,
        str,
        str,
        str,
        int,
        int,
    ]
) -> None:
    """
    Simulate a single run of ECCS user and save the results.

    Parameters:
        args: a tuple containing:
            data: The dataset to be used for causal analysis.
            ground_truth_dag: The ground truth DAG.
            starting_dag: The starting DAG for the ECCS user.
            treatment: The name of the treatment variable.
            outcome: The name of the outcome variable.
            method: The method to be used to suggest edits.
            results_path: The path to save the results to.
            dataset_name: The name of the dataset.
            num_steps: The number of steps to run the ECCS user for.
            i: The index of the run, if applicable.
    """

    (
        data,
        ground_truth_dag,
        starting_dag,
        treatment,
        outcome,
        method,
        results_path,
        dataset_name,
        num_steps,
        i,
    ) = args

    f = open(
        os.path.join(
            results_path,
            "logs",
            f"{dataset_name}_{starting_dag['name']}_{treatment}_{outcome}_{method}{'' if i == None else f'_{i}'}.log",
        ),
        "w",
    )
    sys.stdout = f
    sys.stderr = f

    fixed_list = []
    banned_list = []

    if method.endswith("_oracle"):
        method = method[:-7]
        # Create a fixed list out of edges the ground truth and starting dag share
        fixed_list = [
            (src, dst)
            for src, dst in ground_truth_dag["graph"].edges
            if (src, dst) in starting_dag["graph"].edges
        ]
        # Create a banned list out of edges that are in neither the ground truth nor the starting dag
        all_possible_edges = [
            (src, dst)
            for src in ground_truth_dag["graph"].nodes
            for dst in ground_truth_dag["graph"].nodes
            if src != dst
        ]
        banned_list = [
            (src, dst)
            for src, dst in all_possible_edges
            if (
                ((src, dst) not in ground_truth_dag["graph"].edges)
                and ((src, dst) not in starting_dag["graph"].edges)
            )
        ]

    # Remove edge attributes from starting dag
    for _, __, d in starting_dag["graph"].edges(data=True):
        d.clear()

    try:
        user = ECCSUser(
            data,
            ground_truth_dag["graph"],
            starting_dag["graph"],
            treatment,
            outcome,
            fixed_list,
            banned_list,
        )

        user.run(num_steps, method)
    except:
        traceback.print_exc(file=f)

    f.flush()

    exp_prefix = os.path.join(
        results_path, "data",
        f"{dataset_name}_{starting_dag['name']}_{treatment}_{outcome}_{method}_{'' if i == None else f'{i}_'}",
    )

    np.save(
        f"{exp_prefix}ate_trajectory.npy",
        user.ate_trajectory,
    )
    np.save(
        f"{exp_prefix}ate_diff_trajectory.npy",
        user.ate_diff_trajectory,
    )
    np.save(
        f"{exp_prefix}edit_distance_trajectory.npy",
        user.edit_distance_trajectory,
    )
    np.save(
        f"{exp_prefix}invocation_duration_trajectory.npy",
        user.invocation_duration_trajectory,
    )
    np.save(
        f"{exp_prefix}edits_per_invocation_trajectory.npy",
        user.edits_per_invocation_trajectory,
    )

    f.close()


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path",
        type=str,
        default="./iterative_config.yml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="../../evaluation/",
        help="Output directory path",
    )
    args = parser.parse_args()

    # Read config yml file and create bookkeeping dir structure
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    phases_to_skip = config["general"]["phases_to_skip"]
    path_timestamp = (
        f"{datetime.now()}" if phases_to_skip == 0 else config["general"]["timestamp"]
    )
    work_path = os.path.join(args.out_path, path_timestamp)
    os.makedirs(work_path, exist_ok=True)
    with open(os.path.join(work_path, "config.yml"), "w") as file:
        yaml.dump(config, file)

    # 1. Generate the ground truth dags
    print("-----------------")
    ground_truth_dags_path = os.path.join(work_path, "ground_truth_dags")
    os.makedirs(ground_truth_dags_path, exist_ok=True)
    ground_truth_dags = {}
    if phases_to_skip < 1:
        print(f"{datetime.now()} Phase 1: Generating ground truth dags")
        for _ in tqdm(range(config["gen_dag"]["ground_truth_dags"])):
            ret_dict = RandomDAGGenerator.generate(
                config["gen_dag"]["num_nodes"],
                config["gen_dag"]["edge_prob"],
                tuple(config["gen_dag"]["edge_weight_range"]),
                tuple(config["gen_dag"]["edge_noise_sd_range"]),
                ground_truth_dags_path,
            )
            ground_truth_dags[ret_dict["name"]] = ret_dict
    else:
        print(f"{datetime.now()} Phase 1: Loading ground truth dags")
        for file in tqdm(os.listdir(ground_truth_dags_path)):
            if file.endswith(".dot"):
                name = file[:12]
                ground_truth_dags[name] = {
                    "name": name,
                    "graph": nx.DiGraph(
                        nx.nx_pydot.read_dot(os.path.join(ground_truth_dags_path, file))
                    ),
                    "edge_matrix": np.load(
                        os.path.join(ground_truth_dags_path, f"{name}_edge_matrix.npy")
                    ),
                    "noise_matrix": np.load(
                        os.path.join(ground_truth_dags_path, f"{name}_noise_matrix.npy")
                    ),
                }

    # 2. Generate the datasets
    print("-----------------")
    datasets_path = os.path.join(work_path, "datasets")
    os.makedirs(datasets_path, exist_ok=True)
    dataset_names = {}
    if phases_to_skip < 2:
        print(f"{datetime.now()} Phase 2: Generating datasets")
        for dag_dict in tqdm(ground_truth_dags.values()):
            dataset_names[dag_dict["name"]] = []
            for _ in range(config["gen_dataset"]["datasets_per_ground_truth_dag"]):
                dataset_dict = RandomDatasetGenerator.generate(
                    dag_dict["name"],
                    dag_dict["edge_matrix"],
                    dag_dict["noise_matrix"],
                    config["gen_dataset"]["num_points"],
                    config["gen_dataset"]["min_source_val"],
                    config["gen_dataset"]["max_source_val"],
                    datasets_path,
                )
                dataset_names[dag_dict["name"]].append(dataset_dict["name"])
    else:
        print(f"{datetime.now()} Phase 2: Loading datasets")
        for dag_dict in tqdm(ground_truth_dags.values()):
            dataset_names[dag_dict["name"]] = []
            for file in os.listdir(datasets_path):
                if file.startswith(dag_dict["name"]) and file.endswith(".csv"):
                    dataset_names[dag_dict["name"]].append(file[:29])

    # 3. Generate the random starting dags
    print("-----------------")
    starting_dags_path = os.path.join(work_path, "starting_dags")
    os.makedirs(starting_dags_path, exist_ok=True)
    starting_dags = {}

    if phases_to_skip < 3:
        print(f"{datetime.now()} Phase 3: Generating starting dags")
        for _ in tqdm(range(config["gen_starting_dag"]["starting_dags"])):
            ret_val = RandomDAGGenerator.generate(
                config["gen_starting_dag"]["num_nodes"],
                config["gen_starting_dag"]["edge_prob"],
                tuple(config["gen_starting_dag"]["edge_weight_range"]),
                tuple(config["gen_starting_dag"]["edge_noise_sd_range"]),
                starting_dags_path,
            )
            starting_dags[ret_val["name"]] = ret_val
    else:
        print(f"{datetime.now()} Phase 3: Loading starting dags")
        for file in tqdm(os.listdir(starting_dags_path)):
            if file.endswith(".dot"):
                name = file[:12]
                starting_dags[name] = {
                    "name": name,
                    "graph": nx.DiGraph(
                        nx.nx_pydot.read_dot(os.path.join(starting_dags_path, file))
                    ),
                    "edge_matrix": np.load(
                        os.path.join(starting_dags_path, f"{name}_edge_matrix.npy")
                    ),
                    "noise_matrix": np.load(
                        os.path.join(starting_dags_path, f"{name}_noise_matrix.npy")
                    ),
                }

    # 4. Run the methods and store results
    print("-----------------")
    num_steps = config["run_eccs"]["num_steps"]
    methods = config["run_eccs"]["methods"]
    num_random_tries = config["run_eccs"]["num_tries_for_random_method"]
    num_tasks = len(methods) + (
        (num_random_tries - 1) if "random_single_edge_change" in methods else 0
    )

    if phases_to_skip < 4:
        print(f"{datetime.now()} Phase 4: Starting experiment")
        for method in methods:
            os.makedirs(os.path.join(work_path, method), exist_ok=True)
            os.makedirs(os.path.join(work_path, method, "logs"), exist_ok=True)
            os.makedirs(os.path.join(work_path, method, "data"), exist_ok=True)
        
        tasks = []
        pbar = tqdm(
            total=config["gen_dag"]["ground_truth_dags"]
            * config["gen_dataset"]["datasets_per_ground_truth_dag"]
            * config["gen_starting_dag"]["starting_dags"]
            * int(
                config["gen_dag"]["num_nodes"] * (config["gen_dag"]["num_nodes"] - 1) / 2
            )  # Choices of treatment / outcome
            * num_tasks
        )
        # For each ground truth dag...
        for ground_truth_dag_name, datasets in dataset_names.items():
            ground_truth_dag = ground_truth_dags[ground_truth_dag_name]
            # For each dataset...
            for dataset_name in datasets:
                data = pd.read_csv(os.path.join(datasets_path, f"{dataset_name}.csv"))
                # For each starting dag...
                for starting_dag in starting_dags.values():
                    # For each choice of treatment...
                    for treatment_idx in range(config["gen_dag"]["num_nodes"]):
                        # For each choice of outcome...
                        for outcome_idx in range(
                            treatment_idx + 1, config["gen_dag"]["num_nodes"]
                        ):

                            treatment = f"v{treatment_idx}"
                            outcome = f"v{outcome_idx}"

                            if not nx.has_path(
                                ground_truth_dag["graph"], treatment, outcome
                            ) or not nx.has_path(starting_dag["graph"], treatment, outcome):
                                pbar.update(num_tasks)
                                continue

                            # Run ECCS
                            for method in methods:
                                if method == "random_single_edge_change":
                                    for i in range(num_random_tries):
                                        tasks.append(
                                            (
                                                data,
                                                ground_truth_dag,
                                                starting_dag,
                                                treatment,
                                                outcome,
                                                method,
                                                os.path.join(work_path, method),
                                                dataset_name,
                                                num_steps,
                                                i,
                                            )
                                        )
                                else:
                                    tasks.append(
                                        (
                                            data,
                                            ground_truth_dag,
                                            starting_dag,
                                            treatment,
                                            outcome,
                                            method,
                                            os.path.join(work_path, method),
                                            dataset_name,
                                            num_steps,
                                            None,
                                        )
                                    )

        with ProcessPoolExecutor() as executor:
            # Submit all tasks to the executor
            futures = [executor.submit(simulate, task) for task in tasks]
            handler = partial(kill_still_running, executor, futures)
            signal.signal(signal.SIGINT, handler)

            # As each future completes, update the progress bar
            for _ in as_completed(futures):
                pbar.update(1)
        pbar.close()
    else:
        print(f"{datetime.now()} Phase 4: Skipping experiment")

    # 5. Regenerate plots
    print("-----------------")
    print(f"{datetime.now()} Phase 5: Plotting")
    plotter(work_path, skip=False)


if __name__ == "__main__":

    asyncio.run(main())
