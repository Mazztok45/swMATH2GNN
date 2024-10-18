# swMATH2GNN

This project involves mining swMATH metadata and integrating it into a Graph Neural Network (GNN) using Julia and Python, as part of the research paper "Enhancing Software Detection in Mathematical Research Articles Using Graph Neural Networks and Mathematics Knowledge Graph-Based Citation Modeling".

## Goal

The goal of this project is to use a GNN for link prediction between software and articles to determine the probability that a mathematical paper contains references to specific software.

### Data Extraction

The data was extracted using Julia, as the initial plan was to utilize Julia for the entire project. Metadata was obtained from the swMATH (for software) and zbMATH (for articles) APIs and saved as multiple JSON files. The relevant information was then extracted and saved in a CSV file for further use.

### Statistics and Visualization

Descriptive statistics and visualizations were generated using Julia and can be found in the `notebooks` directory.

### Graph Neural Network (GNN)

The implementation of the GNN was performed in Python due to its advanced capabilities in this area.

In the `hetero_data.py` file, the data is preprocessed into a `HeteroData` object. The nodes represent software and articles, with different types of edges defined as follows:

- **Software-to-software edges**: Based on the `related_software` field in the dataset.
- **Article-to-article edges**: Based on the `references` field in the dataset.
- **Software-to-article edges**: Based on the `standard_article` field in the dataset.

With this information, the data could be trained for link prediction. Link prediction aims to infer the edges that should exist (software-to-article edges) while also considering existing edges that do not require prediction.

Unfortunately, I encountered an issue during training due to an `ImportError` with the `LinkNeighborLoader` in `torch_geometric`. Despite trying suggested solutions from [this article](https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70) and [this GitHub discussion](https://github.com/pyg-team/pytorch_geometric/discussions/7866), I was unable to resolve the issue. Therefore, the link prediction component remains unfinished and can be taken up as a future project.

## Reproducing the Project

To reproduce this project, you need to install Julia. The file `packages.jl` contains all the necessary dependencies. Running `packages.jl` will install all required packages.

### Running the Project

Execute the files in the following order:

1. `GNN_Julia/data_processing.jl` - Data extraction and preprocessing.
2. `GNN_Julia/simplegnn.jl` - Simple GNN implementation.
3. `GNN_Julia/analysis.jl` - Statistical analysis.
4. `GNN/benchmark.py` - Python implementation for benchmarking.

### Update on Dependencies

The `TOML` file contains all necessary dependencies for the project.

---

Feel free to reach out for any questions or contributions to the link prediction task. Any improvements or ideas are welcome!
