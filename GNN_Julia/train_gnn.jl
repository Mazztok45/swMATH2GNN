using CSV
using DataFrames
using GraphNeuralNetworks
using Flux
#using TextAnalysis
#using MultivariateStats
using SparseArrays
using StructTypes
using Graphs
using Combinatorics
#include("hetero_data.jl")
include("../extracting_data/extract_software_metadata.jl")
include("../extracting_data/extract_article_metadata.jl")
#using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe
import .DataReaderszbMATH: extract_articles_metadata
#import MultivariateStats: fit, KernelPCA
import GraphNeuralNetworks: GNNHeteroGraph

software_df  = generate_software_dataframe()

#### FUNCTIONS THAT MUST BE KEPT
#articles_list_dict = extract_articles_metadata()
#node_features = [dic[:msc] for dic in articles_list_dict]
###


### Function to write articles MSC
msc_vec = [join(elem,";") for elem in node_features]
msc_str = join(msc_vec,'\n')
open("msc.txt", "w") do file
    write(file, msc_str)
end
###

### Function to read articles msc
node_features = map(x -> split(x,";"), split(read("msc.txt",String), '\n'))
###

####
collect(combinations(node_features[3], 2))

msc_df = CSV.read("MSC_2020(5).csv",DataFrame)
# Print the column names of the DataFrames to verify
#println("Articles DataFrame columns: ", names(articles_df))
println("Software DataFrame columns: ", names(software_df))

u_soft = unique([dic[:software] for dic in articles_list_dict])

related_soft = [collect(row)[1][2] for row in software_df.related_software]

G = GNNHeteroGraph((:software,:relates_to, :software)=> (software_df.id, related_soft))

using PaddedViews  # for padding functionality

# Determine the maximum size of any array



pad_mat = (x=unique(node_features) .== permutedims(node_features))

GNNHeteroGraph(G, ndata = pad_mat)




# Process the data into a HeteroData() object
#try
#    data = preprocess_heterodata(G, articles_dict, software_df)
#    println("Data preprocessing successful")
#catch e
#    println("Error in preprocess_heterodata: ", e)
#end

# Print data metadata if available
#if isdefined(Main, :data) && data !== nothing
#    metadata = data.metadata
#    println(metadata)
#end

# Define the transformation (RandomLinkSplit equivalent in Julia)
##function random_link_split(data; num_val, num_test, disjoint_train_ratio, neg_sampling_ratio, add_negative_train_samples, edge_types, rev_edge_types)
    # Implement your splitting logic here
    # This is a placeholder function, implement according to your logic
#    return train_data, val_data, test_data
#end

# Transform data into train, validation, and test dataset
#train_data, val_data, test_data = random_link_split(
#    data; 
#    num_val=0.1,
#    num_test=0.2,
#    disjoint_train_ratio=0.3,
#    neg_sampling_ratio=2.0,
#    add_negative_train_samples=false,
#    edge_types=("software", "mentioned_in", "article"),
#    rev_edge_types=("article", "rev_mentioned_in", "software"),
#)

# Define a loader for training data (LinkNeighborLoader equivalent in Julia)#
#function link_neighbor_loader(data, num_neighbors, neg_sampling_ratio, edge_label_index, edge_label, batch_size, shuffle)
    # Implement your loader logic here
    # This is a placeholder function, implement according to your logic
#    return loader
#end

#edge_label_index = train_data["software", "mentioned_in", "article"].edge_label_index
#edge_label = train_data["software", "mentioned_in", "article"].edge_label

#train_loader = link_neighbor_loader(
#    data=train_data,
#    num_neighbors=[20, 10],
#    neg_sampling_ratio=2.0,
#    edge_label_index=(("software", "mentioned_in", "article"), edge_label_index),
#    edge_label=edge_label,
#    batch_size=128,
#    shuffle=true,
#)

# Sample a mini-batch
#function get_sampled_data(loader)
    # Implement your sampling logic here
    # This is a placeholder function, implement according to your logic
#    return sampled_data
#end

#sampled_data = get_sampled_data(train_loader)

#println("Sampled mini-batch:")
#println("===================")
#println(sampled_data)
