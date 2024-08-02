from torch_geometric.loader import LinkNeighborLoader
from hetero_data import preprocess_heterodata
import pandas as pd
import torch_geometric.transforms as T


# Read the dfs into framework
articles_df = pd.read_csv('../articles_metadata_collection/full_df.csv')
articles_dict = articles_df.to_dict()
software_df = pd.read_csv('../data/full_df.csv')
software_dict = software_df.to_dict()

# Process the data into an HeteroData() object
data = preprocess_heterodata(articles_dict=articles_dict, software_dict=software_dict)
metadata = data.metadata()
print(data)


# Transform data into train, validation and test dataset
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("software", "mentioned_in", "article"),
    rev_edge_types=("article", "rev_mentioned_in", "software"),
)

train_data, val_data, test_data = transform(data)

# Use a Loader to train the data

edge_label_index = train_data["software", "mentioned_in", "article"].edge_label_index
edge_label = train_data["software", "mentioned_in", "article"].edge_label

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("software", "mentioned_in", "article"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)


sampled_data = next(iter(train_loader))

print("Sampled mini-batch:")
print("===================")
print(sampled_data)

# The LinkNeighborLoader is the right Loader for this task.
# Unfortunatly, it is not advanced and has some issues that are not easy to solve.
# This is why I stopped working on this approach.
