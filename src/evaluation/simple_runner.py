from __future__ import annotations
import argparse
import yaml
import sys

sys.path.append("../..")
from src.generators.random_dag_generator import RandomDAGGenerator
from src.generators.random_dataset_generator import RandomDatasetGenerator
from src.evaluation.causal_discovery import CausalDiscovery
import networkx as nx
import os
import numpy as np
import pandas as pd
import asyncio



async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path",
        type=str,
        default="./simple_config.yml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="../../datasets/evaluation/",
        help="Output directory path",
    )
    args = parser.parse_args()

    # Read config yml file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # 1. Generate or load the DAG
    dag_name = None
    dag_dict = {}
    if "gen_dag" in config and config["gen_dag"]["enabled"]:
        num_nodes = config["gen_dag"]["num_nodes"]
        edge_prob = config["gen_dag"]["edge_prob"]
        edge_weight_range = tuple(config["gen_dag"]["edge_weight_range"])
        edge_noise_sd_range = tuple(config["gen_dag"]["edge_noise_sd_range"])
        dag_dict = RandomDAGGenerator.generate(
            num_nodes, edge_prob, edge_weight_range, edge_noise_sd_range, args.out_path
        )
    elif "load_dag" in config and config["load_dag"]["enabled"]:
        path = config["load_dag"]["path"]
        dag_name = config["load_dag"]["dag_name"]
        dag_dict = {
            "name": dag_name,
            "graph": nx.DiGraph(
                nx.nx_pydot.read_dot(os.path.join(path, f"{dag_name}_graph.dot"))
            ),
            "edge_matrix": np.load(os.path.join(path, f"{dag_name}_edge_matrix.npy")),
            "noise_matrix": np.load(os.path.join(path, f"{dag_name}_noise_matrix.npy")),
        }
    else:
        raise ValueError("No DAG generation or loading parameters specified")

    if config["exit_after_1"]:
        return

    # 2. Generate or load the dataset
    dataset_name = None
    dataset_dict = {}
    if "gen_dataset" in config and config["gen_dataset"]["enabled"]:
        num_points = config["gen_dataset"]["num_points"]
        min_source_val = config["gen_dataset"]["min_source_val"]
        max_source_val = config["gen_dataset"]["max_source_val"]
        dataset_dict = RandomDatasetGenerator.generate(
            dag_dict["name"],
            dag_dict["edge_matrix"],
            dag_dict["noise_matrix"],
            num_points,
            min_source_val,
            max_source_val,
            args.out_path,
        )
    elif "load_dataset" in config and config["load_dataset"]["enabled"]:
        path = config["load_dataset"]["path"]
        dataset_name = config["load_dataset"]["dataset_name"]
        dataset_dict = {
            "name": dataset_name,
            "data": pd.read_csv(os.path.join(path, f"{dataset_name}.csv")),
        }
    else:
        raise ValueError("No dataset generation or loading parameters specified")

    if config["exit_after_2"]:
        return

    # 3. Generate or load a starting graph generated by a causal discovery algorithm
    starting_graph = None
    starting_graph_name = None
    if "gen_starting_graph" in config and config["gen_starting_graph"]["enabled"]:
        method = config["gen_starting_graph"]["method"]
        option = config["gen_starting_graph"]["option"]
        out_path = config["gen_starting_graph"]["out_path"]
        cd = CausalDiscovery(dataset_dict["name"], dataset_dict["data"])
        starting_graph, _ = await cd.run_with_timer(method, option, out_path)
    elif "load_starting_graph" in config and config["load_starting_graph"]["enabled"]:
        path = config["load_starting_graph"]["path"]
        method = config["load_starting_graph"]["method"]
        option = config["load_starting_graph"]["option"]
        starting_graph = nx.DiGraph(
            nx.nx_pydot.read_dot(
                os.path.join(path, f"{dataset_name}_{method}_{option}.dot")
            )
        )
    elif (
        "random_starting_graph" in config and config["random_starting_graph"]["enabled"]
    ):
        num_nodes = config["random_starting_graph"]["num_nodes"]
        edge_prob = config["random_starting_graph"]["edge_prob"]
        edge_weight_range = tuple(config["random_starting_graph"]["edge_weight_range"])
        edge_noise_sd_range = tuple(
            config["random_starting_graph"]["edge_noise_sd_range"]
        )
        dag_dict = RandomDAGGenerator.generate(
            num_nodes, edge_prob, edge_weight_range, edge_noise_sd_range, args.out_path
        )
        starting_graph = dag_dict["graph"]
        starting_graph_name = dag_dict["name"]
    elif (
        "load_random_starting_graph" in config
        and config["load_random_starting_graph"]["enabled"]
    ):
        path = config["load_random_starting_graph"]["path"]
        starting_graph_name = config["load_random_starting_graph"]["dag_name"]
        starting_graph = nx.DiGraph(
            nx.nx_pydot.read_dot(os.path.join(path, f"{starting_graph_name}_graph.dot"))
        )
    else:
        raise ValueError("No starting graph generation or loading parameters specified")

    if config["exit_after_3"]:
        return

    # 4. Run or load ECCS for the specified number of steps and recover the trajectories.
    print("Data pre-processing done; insert algorithm here.")


if __name__ == "__main__":
    asyncio.run(main())
