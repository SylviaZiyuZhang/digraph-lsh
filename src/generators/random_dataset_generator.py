import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import json
import os


class RandomDatasetGenerator:
    """
    A class for creating a random dataset based on an input causal DAG.
    """

    DEFAULT_NUM_POINTS = 1000
    DEFAULT_MIN_SOURCE_VAL = 0
    DEFUALT_MAX_SOURCE_VAL = 10

    @staticmethod
    def generate(
        dag_name: str,
        edge_matrix: np.ndarray,
        noise_matrix: np.ndarray,
        num_points: int = DEFAULT_NUM_POINTS,
        min_source_val: float = DEFAULT_MIN_SOURCE_VAL,
        max_source_val: float = DEFUALT_MAX_SOURCE_VAL,
        out_path: str = None,
    ) -> dict[str, object]:
        """
        Generates a random dataset of `num_points` data points based on the given matrices about the causal DAG.
        We assume that the nodes in the DAG are in topological order.

        Parameters:
            dag_name: The name of the DAG.
            edge_matrix: The edge weight matrix of the causal DAG.
            noise_matrix: The noise standard deviation matrix of the causal DAG.
            num_points: The number of data points.
            min_source_val: The minimum value for source variables.
            max_source_val: The maximum value for source variables.
            out_path: The path to write the dataset to. If None, the dataset is not written out.

        Returns:
            A dictionary with the following elements:
            - name: The name of the generated dataset.
            - data: The generated dataset.
        """

        num_vars = edge_matrix.shape[0]

        data = np.zeros((num_points, num_vars))
        data[:, 0] = np.random.uniform(min_source_val, max_source_val, num_points)
        for j in range(1, num_vars):
            # Determine that this variable is a source if all incoming weights are zero.
            if np.all(edge_matrix[:, j] == 0):
                data[:, j] = np.random.uniform(
                    min_source_val, max_source_val, num_points
                )
                continue

            # Calculate the value of the jth variable based on the values of the previous variables (incoming edges).
            for i in range(j):
                data[:, j] += edge_matrix[i, j] * data[:, i] + np.random.normal(
                    0, noise_matrix[i, j]
                )

        # Convert to dataframe
        data_df = pd.DataFrame(data, columns=[f"v{i}" for i in range(num_vars)])

        # Write out optionally
        dataset_name = f"{dag_name}_dataset_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        if out_path:
            # Generation parameters
            generation_params = {
                "num_points": num_points,
                "min_source_val": min_source_val,
                "max_source_val": max_source_val,
            }
            with open(
                os.path.join(out_path, f"{dataset_name}_parameters.json"), "w"
            ) as f:
                json.dump(generation_params, f)

            data_df.to_csv(os.path.join(out_path, f"{dataset_name}.csv"), index=False)

        return {"name": dataset_name, "data": data_df}
