import argparse
import numpy as np
import datetime
import json
import pandas as pd


DEFAULT_ARGS = {
    "length": 1000,
    "num_total_variables": 100,
    "noise_radius": 1,
}


def xyz_extended_gen(args):
    ts = datetime.datetime.utcnow()
    filename = f"../../datasets/xyz_extended/dataset_{ts.year:04d}-{ts.month:02d}-{ts.day:02d}_{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"

    # Dump arguments
    with open(filename + ".json", "w+") as f:
        json.dump(args, f, indent=2)

    column_names = ["x", "y", "z"] + [
        f"var_{i}" for i in range(args["num_total_variables"] - 3)
    ]

    # Compose the csv
    data_filename = filename + ".csv"
    with open(data_filename, "a+") as f:

        z = np.random.uniform(0, 100, (args["length"], 1))

        x = z + np.random.normal(0, args["noise_radius"], (args["length"], 1))

        y = (
            3 * x
            + 2 * z
            + np.random.normal(0, args["noise_radius"], (args["length"], 1))
        )

        # Draw the remaining values
        rest = np.random.random((args["length"], args["num_total_variables"] - 3)) * 100

        # Concatenate the values and cast to dataframe
        data = np.concatenate((x, y, z, rest), axis=1)
        df = pd.DataFrame(data, columns=column_names)

        # Write to file
        df.to_csv(f, index=False)

    # Compose the graph files
    simple_graph_filename = filename + "_simple.dot"
    with open(simple_graph_filename, "w+") as f:
        f.write("digraph G {\n")
        f.write("x -> y;\n")
        f.write("}\n")

    correct_graph_filename = filename + "_correct.dot"
    with open(correct_graph_filename, "w+") as f:
        f.write("digraph G {\n")
        f.write("x -> y;\n")
        f.write("z -> x;\n")
        f.write("z -> y;\n")
        f.write("}\n")

    incorrect_graph_filename = filename + "_incorrect.dot"
    with open(incorrect_graph_filename, "w+") as f:
        v = np.random.randint(0, args["num_total_variables"] - 3)
        f.write("digraph G {\n")
        f.write("x -> y;\n")
        f.write(f"var_{v} -> x;\n")
        f.write(f"var_{v} -> y;\n")
        f.write("}\n")

    return data_filename


if __name__ == "__main__":
    # Accept arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--num_total_variables", type=int, default=100)
    parser.add_argument("--noise_radius", type=float, default=1)
    args = parser.parse_args().__dict__
    xyz_extended_gen(args)
