# digraph-lsh
Artifact for the final project on locality sensitive hashing for directed graphs in 6.5320 Geometric Computing at MIT.


# Synthetic Experiments

Reproducing synthetic experiments is simple and can be done by running 

```
python3 synthetic_experiment.py
```


# Pythia Circuits Experiments

Reproducing these experiment is more involved. First, install git lfs, then run in a different folder

```
git clone https://github.com/JoshEngels/feature-circuits-clusters
cd feature-circuits-clusters
git lfs install
git lfs pull
python3 extract_cluster_dotfiles.py
python3 save_contexts.py
```

You can then cd back to this folder and run

```
ln -s /path/to/feature-circuits-clusters/dotfiles dotfiles
```

and finally run the experiment:

```
python3 nn_experiment.py
```