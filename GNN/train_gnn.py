from build_gnn import metadata, train_data, val_data, test_data, model, data
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.typing


print(data.validate())
print("running 1")
edge_label_index = train_data["software", "mentioned_in", "article"].edge_label_index
edge_label = train_data["software", "mentioned_in", "article"].edge_label
print("running 2")
print(torch_geometric.typing.WITH_PYG_LIB)

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("software", "mentioned_in", "article"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)
print("running 3")

# Inspect a sample:
sampled_data = next(iter(train_loader))

print("Sampled mini-batch:")
print("===================")
print(sampled_data)
