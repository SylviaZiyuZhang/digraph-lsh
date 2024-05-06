import networkx as nx
import random
import numpy as np
from datetime import datetime
import json
import hashlib
import os
from ..utils.graph_renderer import GraphRenderer
from ..utils.edge_state_matrix import EdgeStateMatrix, EdgeState


class RandomDAGGenerator:
    """
    A class for the creation of random DAGs, with a potential causal interpretation.
    """

    DEFAULT_NUM_NODES = 10
    DEFAULT_EDGE_PROB = 0.25
    DEFAULT_EDGE_WEIGHT_RANGE = (-10, 10)
    DEFAULT_EDGE_NOISE_SD_RANGE = (-2, 2)

    @classmethod
    def generate(
        cls,
        num_nodes: int = DEFAULT_NUM_NODES,
        edge_prob: float = DEFAULT_EDGE_PROB,
        edge_weight_range: tuple[float, float] = DEFAULT_EDGE_WEIGHT_RANGE,
        edge_noise_sd_range: tuple[float, float] = DEFAULT_EDGE_NOISE_SD_RANGE,
        out_path: str = None,
    ) -> dict[str, object]:
        """
        Generates a random DAG pased on the provided parameters.

        Parameters:
            num_nodes: The number of nodes in the DAG.
            edge_prob: The probability of an edge existing between any two nodes.
            edge_weight_range: The range of edge weights.
            edge_noise_sd_range: The range of edge noise standard deviations.
            out_path: The path to write the DAG to. If None, the DAG is not written out.

        Returns:
            A dictionary with the following elements:
            - name: The name of the generated DAG.
            - graph: The generated DAG.
            - edge_matrix: The edge weights of the generated DAG.
            - noise_matrix: The noise standard deviations of the generated DAG.
        """

        # Create a random DAG. Order nodes and only allow "forward" edges to ensure DAG.
        G = nx.DiGraph()
        for i in range(num_nodes):
            G.add_node(f"v{i}", var_name=f"v{i}")

        edge_matrix = np.array(
            [
                [
                    (
                        0
                        if i >= j
                        else (
                            random.uniform(*edge_weight_range)
                            if random.random() < edge_prob
                            else 0
                        )
                    )
                    for j in range(num_nodes)
                ]
                for i in range(num_nodes)
            ]
        )

        noise_matrix = np.array(
            [
                [
                    (
                        0
                        if edge_matrix[i][j] == 0
                        else random.uniform(*edge_noise_sd_range)
                    )
                    for j in range(num_nodes)
                ]
                for i in range(num_nodes)
            ]
        )

        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if edge_matrix[i, j] != 0:
                    G.add_edge(
                        f"v{i}",
                        f"v{j}",
                        weight=edge_matrix[i, j],
                        noise_sd=noise_matrix[i, j],
                    )

        # Write out optionally

        # hash the current timestamp to generate a dataset name
        dag_name = f"dag_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        if out_path:
            # Generation parameters
            generation_params = {
                "num_nodes": num_nodes,
                "edge_prob": edge_prob,
                "edge_weight_range": edge_weight_range,
                "edge_noise_sd_range": edge_noise_sd_range,
            }
            with open(os.path.join(out_path, f"{dag_name}_parameters.json"), "w") as f:
                json.dump(generation_params, f)

            # Generated Products
            nx.nx_pydot.write_dot(G, os.path.join(out_path, f"{dag_name}_graph.dot"))
            np.save(os.path.join(out_path, f"{dag_name}_edge_matrix.npy"), edge_matrix)
            np.save(
                os.path.join(out_path, f"{dag_name}_noise_matrix.npy"), noise_matrix
            )
            esm = EdgeStateMatrix([f"v{i}" for i in range(num_nodes)])
            for src, dst in G.edges():
                esm.mark_edge(src, dst, EdgeState.PRESENT)
            GraphRenderer.save_graph(
                G,
                esm,
                os.path.join(out_path, f"{dag_name}_graph.png"),
            )

        out_dict = {
            "name": dag_name,
            "graph": G,
            "edge_matrix": edge_matrix,
            "noise_matrix": noise_matrix,
        }

        return out_dict
