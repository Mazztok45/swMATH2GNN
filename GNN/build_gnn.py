from hetero_data import preprocess_heterodata
import pandas as pd
import torch
from torch.nn import Linear, Module
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv
import torch_geometric.transforms as T


articles_df = pd.read_csv('../articles_data/full_df.csv')
articles_dict = articles_df.to_dict()
software_df = pd.read_csv('../data/full_df.csv')
software_dict = software_df.to_dict()

data = preprocess_heterodata(articles_dict=articles_dict, software_dict=software_dict)

print(data)


class HeteroGNN(Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super(HeteroGNN, self).__init__()
        self.convs = HeteroConv({
            ('article', 'references', 'article'): GCNConv(-1, hidden_channels),
            ('software', 'related', 'software'): GCNConv(-1, hidden_channels),
            ('software', 'mentioned_in', 'article'): SAGEConv((-1, -1), hidden_channels)
        }, aggr='mean')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        x_dict = self.convs(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

    def encode(self, x_dict, edge_index_dict, edge_weight_dict=None):
        return self.forward(x_dict, edge_index_dict, edge_weight_dict)

    def decode(self, z_dict, edge_label_index):
        source, target = edge_label_index
        return (z_dict['software'][source] * z_dict['article'][target]).sum(dim=-1)


transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("software", "mentioned_in", "article"),
    rev_edge_types=("article", "rev_mentioned_in", "software"),
)



metadata = data.metadata()
model = HeteroGNN(metadata, hidden_channels=32, out_channels=32)
print(model)
train_data, val_data, test_data = transform(data)
print("Trainig: ", train_data)
print("val: ", val_data)
print("test: ", test_data)


