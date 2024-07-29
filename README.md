# swMATH2GNN
This work is about mining the swMATH metadata and ingest them into Graph Neural Networks with libraries written in Julia and Python.

## Goal
The Goal of this work is to use the GNN for link prediction between software and articles to conlcude the
probability that a mathematical paper contains a certain software. 

Firstly, we need extract the metadata

### Data extraction

The data was extracted in Julia, because the originla plan was to use Julia for this whole project.
The data was downloaded from th swMath (for the software) and zbMath (for the articles) API and saved as
multiple json files. The information needed was extracted using julia and saved in a 
csv file for further use

### Statistics and Visualisation
Some descriptive Statistics and visualistion (also with julia) can be found in the notebooks directory

### Software network
I build a simple network for the  relations between the software in Julia. It wasn't used for the GNN 
in the end, because I switched to python

### GNN
In hetero_data.py I preprocessed the data into a HeteroData object.
The nodes are software and 


