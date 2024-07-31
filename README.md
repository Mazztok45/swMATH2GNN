# swMATH2GNN
This work is about mining the swMATH metadata and ingest them into Graph Neural Network with Julia and Python.

## Goal
The Goal of this work is to use the GNN for link prediction between software and articles to conlcude the
probability that a mathematical paper contains a certain software. 

Firstly, we need to extract the metadata

### Data extraction

The data was extracted in Julia, because the originla plan was to use Julia for this whole project.
The data was downloaded from th swMath (for the software) and zbMath (for the articles) API and saved as
multiple json files. The information needed was extracted using julia and saved in a 
csv file for further use

### Statistics and Visualisation
Some descriptive Statistics and visualistion (also with julia) can be found in the notebooks directory.

### GNN
I decided to work in python for this task, because it is more advanced in this topic.

In hetero_data.py I preprocessed the data into a HeteroData object.
The nodes are software and article.
There are edges between the software nodes, based on the 'related_software' information in the dataset.
There are edges between the article nodes, based on the 'references' information in the dataset.
There are edges between the software nodes and article nodes, based on the 'standart_article' information
in the dataset.

With this Information, the data could be trained with link prediction. Link prediction takes the edges that are supposed 
to be predicted into account (the software to article edges), but also the edges between the nodes that don't need perdiction. 

Unfortunatly, I failed to train the data due to an ImportError in the LinkNeighborLoader 
in torch_geometric that I could not solve. The LinkNeighborLoader is the right training data loader 
for this task, according to [this article](https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70).
I also found a [discussion of the problem](https://github.com/pyg-team/pytorch_geometric/discussions/7866)
in the GitHub of PyG. But the suggested solutions did not work for me.

The link prediction part of this project will have to be a future project for someone else.

## Information on julia
To reproduce this project, julia need to be installed. The file packages.jl includes all 
packages that need to be installed. When the packages.jl file is run, all packages are installed.




