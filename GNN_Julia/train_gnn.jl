using CSV
using DataFrames
using GraphNeuralNetworks
using Flux
using TextAnalysis
using MultivariateStats
using SparseArrays
using StructTypes
using TikzGraphs
using Graphs
using GraphPlot
using GraphRecipes
using Compose
include("hetero_data.jl")
include("../extracting_data/extract_software_metadata.jl")
using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe
import MultivariateStats: fit, KernelPCA
import GraphNeuralNetworks: GNNHeteroGraph
import Cairo
import Fontconfig

# Load full data into DataFrames
articles_df = CSV.read("./articles_metadata_collection/full_df.csv", DataFrame)
# Function to parse the `related_software` JSON strings


# Read the CSV file using the custom struct
#software_df = CSV.read("./data/full_df.csv", DataFrame; types=Dict(:related_software => String))

software_df  = generate_software_dataframe()
# Print the column names of the DataFrames to verify
println("Articles DataFrame columns: ", names(articles_df))
println("Software DataFrame columns: ", names(software_df))

# Create Article instances
#articles_dict = Dict{String, Article}()
#for i in 1:nrow(articles_df)
#    row = articles_df[i, :]
#    article_instance = create_article(row)
#    articles_dict[string(row[:id])] = article_instance
#end

# Create Software instances
#software_dict = Dict{String, Software}()
#for i in 1:nrow(software_df)
#    row = software_df[i, :]
#    software_instance = create_software(row)
    #println(software_instance)
#    software_dict[string(row[:id])] = software_instance
#end

# Print keys to verify dictionaries
#println("Articles Dict Keys: ", keys(articles_dict))
#println("Software Dict Keys: ", keys(software_dict))

related_soft = [collect(row)[1][2] for row in software_df.related_software]

G = GNNHeteroGraph((:software,:relates_to, :software)=> (software_df.id, related_soft))

##### VIZ PART ###
filtered_df = software_df#[software_df.articles_count .> 200, :]
related_soft_filtered_df = [collect(row)[1][2] for row in filtered_df.related_software]
g_viz=DiGraph()
add_vertices!(g_viz, length(Set(filtered_df.id)))
for i in 1:nrow(filtered_df)
    add_edge!(g_viz,filtered_df[i, :id],related_soft_filtered_df[i])
end

sg = induced_subgraph(g_viz, neighbors(g_viz,825))


vertices_l = reduce(hcat, [[src(e), dst(e)] for e in edges(sg[1])])
g_viz2=DiGraph()
add_vertices!(g_viz2, length(Set(vertices_l)))
for edge in edges(sg[1])
    add_edge!(g_viz2, src(edge), dst(edge))
end




l_s = unique(software_df[!,[:name,:id]])

nodelabel = [l_s[l_s.id  .== id_s,:].name[1] for id_s in sg[2]]
nodesize = fill(5, 20)

draw(PNG("soft_825_graph.png", 50cm, 50cm), gplot(g_viz2, nodelabel=nodelabel, nodesize=nodesize))
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
