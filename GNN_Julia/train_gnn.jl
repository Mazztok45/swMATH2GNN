include("hetero_data.jl")
using CSV
using DataFrames
using GraphNeuralNetworks
using Flux
using .HeteroData

# Load data into DataFrames
articles_df = CSV.read("./articles_metadata_collection/full_df.csv", DataFrame)
software_df = CSV.read("./data/full_df.csv", DataFrame)

# Convert DataFrames to dictionaries
articles_dict = Dict(name => articles_df[!, name] for name in names(articles_df))
software_dict = Dict(name => software_df[!, name] for name in names(software_df))

# Define a function to preprocess heterogenous data


# Process the data into an HeteroData() object
data = preprocess_heterodata(articles_dict, software_dict)

# Print data metadata
metadata = data.metadata
#println(data)

# Define the transformation (RandomLinkSplit equivalent in Julia)
function random_link_split(data; num_val, num_test, disjoint_train_ratio, neg_sampling_ratio, add_negative_train_samples, edge_types, rev_edge_types)
    # Implement your splitting logic here
    # This is a placeholder function, implement according to your logic
    return train_data, val_data, test_data
end

# Transform data into train, validation and test dataset
train_data, val_data, test_data = random_link_split(
    data; 
    num_val=0.1,
    num_test=0.2,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=false,
    edge_types=("software", "mentioned_in", "article"),
    rev_edge_types=("article", "rev_mentioned_in", "software"),
)

# Define a loader for training data (LinkNeighborLoader equivalent in Julia)
function link_neighbor_loader(data, num_neighbors, neg_sampling_ratio, edge_label_index, edge_label, batch_size, shuffle)
    # Implement your loader logic here
    # This is a placeholder function, implement according to your logic
    return loader
end

edge_label_index = train_data["software", "mentioned_in", "article"].edge_label_index
edge_label = train_data["software", "mentioned_in", "article"].edge_label

train_loader = link_neighbor_loader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("software", "mentioned_in", "article"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=true,
)

# Sample a mini-batch
function get_sampled_data(loader)
    # Implement your sampling logic here
    # This is a placeholder function, implement according to your logic
    return sampled_data
end


#sampled_data = get_sampled_data(train_loader)

println("Sampled mini-batch:")
println("===================")
#println(sampled_data)
