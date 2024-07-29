from hetero_data import preprocess_heterodata
from build_gnn import HeteroGNN
import torch_geometric.transforms as T


transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("software", "mentioned_in", "article"),
)