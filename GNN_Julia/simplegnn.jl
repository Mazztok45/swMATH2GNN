# An example of semi-supervised node classification
using Flux
using Flux: onecold, onehotbatch, onehot
using Flux.Losses: logitcrossentropy
using GraphNeuralNetworks
using MLDatasets: Cora
using Statistics, Random
using CUDA
CUDA.allowscalar(false)
using CSV
using DataFrames

using Random
#using TextAnalysis
using MultivariateStats
using SparseArrays
using StructTypes
using Graphs
#using Combinatorics
#include("hetero_data.jl")
include("../extracting_data/extract_software_metadata.jl")
include("../extracting_data/extract_article_metadata.jl")
#using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe
import .DataReaderszbMATH: extract_articles_metadata
#import MultivariateStats: fit, KernelPCA
import GraphNeuralNetworks: GNNHeteroGraph, GNNGraphs
#using OneHotArrays
using MLLabelUtils
using StatsBase
using Arrow
using Serialization
#using NearestNeighbors
using JLD2
using Metis
using BSON
#using Node2Vec
using LinearAlgebra
using IterativeSolvers
using HTTP
using JSON
using MLUtils



function msc_encoding()
    #return DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow"))
    return deserialize("dense_one_hot.jls")
end

#msc_features_np=Matrix{Float32}(msc_encoding())


pi_int = parse.(Int, split(read("msc_paper_id.txt", String), '\n'))
unique_paper_ids = Set(unique(pi_int))

function paper_edges()
    return DataFrame(Arrow.Table("GNN_Julia/papers_edges_arrow/papers_edges_arrow"))
end




filtered_data = filter(row -> row.paper_id in unique_paper_ids, unique(paper_edges()[!, [:paper_id, :software]]))

grouped_by_paper_id = combine(groupby(filtered_data, :paper_id), :software => x -> collect(x) => :software_array)


# Extract the software arrays from the first element of each pair
software_arrays = [pair.first for pair in grouped_by_paper_id.software_function]

# Cleaning the memory
pi_int = nothing
unique_paper_ids = nothing
filtered_data = nothing
grouped_by_paper_id = nothing

GC.gc()

# Collect all unique software labels
all_software_labels = unique(vcat(software_arrays...))

# Map each software label to a unique index
label_to_index = Dict(label => i for (i, label) in enumerate(all_software_labels))


# Number of unique software labels
num_labels = length(all_software_labels)


# Loop through each array of software labels in software_arrays
#for software_list in software_arrays
# Create a zero vector of length equal to the number of unique labels
#    one_hot_vector = zeros(Int, num_labels)

# For each software in the list, set the corresponding index in the one-hot vector to 1
#    for software in software_list
#        idx = label_to_index[software]
#        one_hot_vector[idx] = 1
#    end

# Append the multi-hot encoded vector to the list
#    push!(multi_hot_encoded, one_hot_vector)
#end

#######
# Initialize a BitArray with the dimensions (num_labels × number of papers)
num_papers = length(software_arrays)

# Create an uninitialized BitArray (BitMatrix)
multi_hot_matrix = BitArray(undef, num_labels, num_papers)

# Loop through each array of software labels in software_arrays
for (i, software_list) in enumerate(software_arrays)
    # For each software in the list, set the corresponding index in the BitArray to true (1)
    for software in software_list
        idx = label_to_index[software]
        multi_hot_matrix[idx, i] = true  # Set the bit to true
    end
end

multi_hot_matrix = permutedims(multi_hot_matrix)

#multi_label_data = [map(x -> label_to_index[x],l) for l in software_arrays]
software_df  = generate_software_dataframe()
u_soft=unique(software_df[!,[:id,:classification]])
vs=replace!(u_soft[!,:classification],"" => "No_MSC")
u_soft[!,:classification]=vs

# Create a Set from u_soft.id for fast membership checking
existing_ids = Set(u_soft.id)

# Replace missing classification with "No_MSC"
replace!(u_soft[!,:classification], "" => "No_MSC")

# Prepare to accumulate new data for u_soft
new_entries = Vector{Tuple{Int64, String}}()

# Iterate over software arrays
for l in software_arrays
    for i in l
        int_id = parse(Int64, i)

        # If int_id is not in existing_ids, make a GET request
        if !(int_id in existing_ids)
            req = "https://api.zbmath.org/v1/software/" * string(i)
            println(req)
            
            # Perform the HTTP request
            response = HTTP.get(req)
            data = JSON.parse(String(response.body))

            # Append the new (int_id, classification) to new_entries
            push!(new_entries, (int_id, join(data["result"]["classification"], ";")))

            # Add the int_id to existing_ids to avoid duplicate requests
            push!(existing_ids, int_id)
        end    
    end
end

# Finally, append new entries to the u_soft DataFrame
if !isempty(new_entries)
    append!(u_soft, DataFrame(new_entries, [:id, :classification]))
end

# Parse the response body

# Precompute a lookup table for u_soft by creating a dictionary from id to classification
id_to_classification = Dict(u_soft.id .=> u_soft.classification)

# Initialize result vector
vec_msc_soft = Vector{Vector}()

# Iterate over software arrays
for l in software_arrays
    arr = Vector{String}()

    # Collect classification values based on id
    for i in l
        if haskey(id_to_classification, parse(Int64, i))
            # Split classification string into MSC codes and append
            append!(arr, split(id_to_classification[parse(Int64, i)], ';'))
        end
    end

    # Push unique MSC codes to the result vector
    push!(vec_msc_soft, unique(arr))
end


# Initialize the number of papers and MSC labels
num_papers = length(software_arrays)

# Get all unique MSC labels and map them to unique indices
all_software_msc = unique(vcat(vec_msc_soft...))
label_to_msc = Dict(label => i for (i, label) in enumerate(all_software_msc))

num_msc = length(all_software_msc)

# Pre-allocate the BitArray (BitMatrix) to hold the hot-encoded data
msc_soft_hot_matrix = falses(num_msc, num_papers)  # Initialize with all false (0)

# Populate the BitMatrix by setting the corresponding bits to true (1)
for (i, msc_list) in enumerate(vec_msc_soft)
    for msc in msc_list
        idx = label_to_msc[msc]
        msc_soft_hot_matrix[idx, i] = true
    end
end

msc_soft_hot_matrix= permutedims(msc_soft_hot_matrix)
serialize("msc_soft_hot_matrix.jls",msc_soft_hot_matrix)
##############


# Number of unique labels (30,694) and papers (146,346)
#num_labels = 30694
#num_papers = length(multi_label_data)

# Function to handle both single-label and multi-label cases without losing OneHotMatrix type
#function combine_labels(labels::Vector{Int64}, num_labels::Int64)
# Ensure all labels are within the valid range
#    if all(l -> l in 1:num_labels, labels)
#        if length(labels) == 1
# Single label: simply return the OneHotVector for that label
#            return Flux.onehot(labels[1], 1:num_labels)
#        else
# Multi-label case: We generate multiple OneHotVectors and sum them
#            return sum(Flux.onehotbatch(labels, 1:num_labels), dims=2)
#        end
#    else
#        error("One or more labels are out of the valid range 1 to $num_labels")
#    end
#end

# Apply the function to each paper's labels and collect the one-hot encoded vectors
#one_hot_vectors = [combine_labels(labels, num_labels) for labels in multi_label_data]

# Use reduce with hcat for efficient concatenation, preserving OneHotMatrix type


filtered_data = nothing
grouped_by_paper_id = nothing
multi_label_data = nothing
multi_hot_matrix = nothing

GC.gc()


final_one_hot_matrix = reduce(hcat, one_hot_vectors)

println("Size of final OneHotMatrix: ", size(final_one_hot_matrix))



##############



############


# Number of unique labels (30,694) and papers (146,346)
num_labels = 30694
num_papers = length(multi_label_data)

# Preallocate a dense matrix to store one-hot encodings directly
one_hot_data = zeros(UInt32, num_labels, num_papers)






#filtered_data = filter(row -> row.paper_id in unique_paper_ids, unique(paper_edges()[!,[:paper_id,:software]]))

#grouped_by_paper_id = combine(groupby(filtered_data, :paper_id), :software => x -> collect(x) => :software_array)



# Loop through each paper, encode its labels, and fill the preallocated matrix
#for i in 1:num_papers
#    labels = multi_label_data[i]
#    one_hot_vector = combine_labels(labels, num_labels)
#    one_hot_data[:, i] = one_hot_vector  # Directly assign the one-hot vector to the matrix column

# Run GC every 10,000 iterations instead of every loop
#    if i % 10000 == 0
#        GC.gc()
#    end
#end

#println("Size of final matrix: ", size(one_hot_data))

############



serialize("multi_hot_matrix.jls", multi_hot_matrix)
#Arrow.write("GNN_Julia/multi_hot_encoded.csv", DataFrame(Matrix{Bool}(permutedims(multi_hot_matrix)), :auto))
# Display one of the multi-hot encoded vectors for checking
#println(multi_hot_encoded[1])


selected_paper_id = Set(unique(grouped_by_paper_id.paper_id))

sd = setdiff(unique_paper_ids, selected_paper_id)

vec_u = collect(unique_paper_ids)


l_ind = [i for (i, j) in enumerate(vec_u) if !(j in sd)]

filtered_msc = permutedims(msc_encoding()[l_ind, :])

serialize("filtered_msc.jls", filtered_msc)

#Arrow.write("GNN_Julia/filtered_msc",filtered_msc)





### ADAPT THE EDGES
refs_df = paper_edges()

filtered_edges = filter(row -> (row.paper_id in unique_paper_ids && !(row.paper_id in sd)), unique(paper_edges()[!, [:paper_id, :ref_id]]))

# Assuming l is a list of ref_ids
l = unique(filtered_edges.paper_id)
# Convert l to a Set for efficient membership checking
l_set = Set(l)
# Create refs_id2 by iterating through each row
filtered_edges.refs_id2 = [row.ref_id in l_set ? row.ref_id : row.paper_id for row in eachrow(filtered_edges)]



Arrow.write("GNN_Julia/filtered_edges", filtered_edges)


######## SPLITING MASKS



#filtered_edges= DataFrame(Arrow.Table(Arrow.read("GNN_Julia/filtered_edges")))


function random_mask(multi_hot_matrix, at=0.7, eval_ratio=0.15)
    n = size(multi_hot_matrix, 1)  # Total number of samples
    num_labels = size(multi_hot_matrix, 2)  # Number of labels (software)

    # Initialize boolean vectors for train, eval, and test masks
    train_mask = BitVector(zeros(Bool, n))
    eval_mask = BitVector(zeros(Bool, n))
    test_mask = BitVector(zeros(Bool, n))

    # Stratified sampling based on label presence in multi_hot_matrix
    for i in 1:num_labels
        # Find indices where the current label is present
        label_indices = findall(multi_hot_matrix[:, i] .== 1)
        shuffle!(label_indices)  # Shuffle indices to randomize

        # Determine the number of samples for train, eval, and test
        num_train = floor(Int, at * length(label_indices))
        num_eval = floor(Int, eval_ratio * length(label_indices))
        num_test = length(label_indices) - num_train - num_eval

        # Assign masks based on stratified sampling
        train_mask[label_indices[1:num_train]] .= true
        eval_mask[label_indices[(num_train+1):(num_train+num_eval)]] .= true
        test_mask[label_indices[(num_train+num_eval+1):end]] .= true
    end

    return train_mask, eval_mask, test_mask
end

# Generate the masks with stratified sampling based on multi_hot_matrix
train_mask, eval_mask, test_mask = random_mask(msc_soft_hot_matrix)

# Save the masks
serialize("train_mask.jls", train_mask)
serialize("eval_mask.jls", eval_mask)
serialize("test_mask.jls", test_mask)




#Arrow.write("GNN_Julia/random_mask", DataFrame(train_mask=train_mask,test_mask=test_mask))




#### MODEL PART


msc_soft_hot_matrix=deserialize("msc_soft_hot_matrix.jls")
#multi_hot_matrix = deserialize("multi_hot_matrix.jls")
filtered_edges = DataFrame(Arrow.Table(Arrow.read("GNN_Julia/filtered_edges")))
filtered_msc = deserialize("filtered_msc.jls")
train_mask = deserialize("train_mask.jls")
eval_mask = deserialize("eval_mask.jls")
test_mask = deserialize("test_mask.jls")


############################ Target preparation

# Assuming multi_hot_matrix is sparse
sparse_matrix_float = SparseMatrixCSC{Float64,Int}(multi_hot_matrix)
# Perform truncated SVD using svdl
S, factorization = svdl(sparse_matrix_float, nsv=50)
# 'P' contains the left singular vectors (U) from the SVD, 
# representing the reduced-dimensional features for each paper, 
# aligned with the rows of the original multi-label matrix.
P = Float32.(permutedims(factorization.P[:, 1:100]))
Q = Float32.(permutedims(factorization.Q[:, 1:100]))

# Sum the multi_hot_matrix along the columns to get label distribution
label_counts = sum(multi_hot_matrix, dims=1)

# Calculate class weights (inverse of the label counts)
class_weights = 1.0 ./ label_counts

############################ Target Preparation

# Assuming multi_hot_matrix is sparse
sparse_matrix_float = SparseMatrixCSC{Float64,Int}(multi_hot_matrix)

# Sum the multi_hot_matrix along the columns to get label distribution
label_counts = sum(multi_hot_matrix, dims=1)

# Calculate class weights (inverse of the label counts)
class_weights = SparseMatrixCSC{Float64, Int}(1.0 ./ label_counts)

# Apply class weights in a loop to avoid memory overflow
weighted_multi_hot_matrix = copy(sparse_matrix_float)  # Initialize a copy

# Multiply each column of the multi-hot matrix by the corresponding class weight
for col in 1:size(sparse_matrix_float, 2)
    weighted_multi_hot_matrix[:, col] .= sparse_matrix_float[:, col] .* class_weights[col]
end

# Perform truncated SVD using svdl on the weighted matrix
S, factorization = svdl(weighted_multi_hot_matrix, nsv=50)

# 'P' contains the left singular vectors (U) from the SVD, 
# representing the reduced-dimensional features for each paper, 
# aligned with the rows of the original multi-label matrix.
P = Float32.(permutedims(factorization.P[:, 1:100]))
Q = Float32.(permutedims(factorization.Q[:, 1:100]))



############################ 


# Parameters for the matrix
#num_papers = 146346  # number of papers (rows)



#num_nodes = Dict(:paper => num_papers)

data = ((:paper,:cited_by,:paper)=>(Vector(filtered_edges.refs_id2), Vector(filtered_edges.paper_id)))

data = unique(DataFrame(hcat(Vector(filtered_edges.refs_id2), Vector(filtered_edges.paper_id)), :auto))


# Combine all node IDs and find unique ones
all_nodes = unique(vcat(data.x1, data.x2))

# Create a dictionary to map old node IDs to new sequential IDs
node_map = Dict(node => i for (i, node) in enumerate(all_nodes))

# Remap the node IDs in your data
new_x1 = [node_map[node] for node in data.x1]
new_x2 = [node_map[node] for node in data.x2]

# Create a new GNNGraph with remapped node IDs




############################ Apply PCA
if isfile("pca_model.jld2")
    @load "pca_model.jld2" pca_model
else
    k = 100  # Number of principal components you want to keep
    pca_model = fit(PCA, filtered_msc; maxoutdim=k)
    @save "pca_model.jld2" pca_model
end

# Transform the data to the new reduced space
reduced_data = predict(pca_model, filtered_msc)
############################


############################ KNN (not working)

#= # Step 2: Build KNN graph using PCA features
if isfile("knn_tree.jld2")
    @load "knn_tree.jld2" knn_tree
else
    k = 10  # Number of nearest neighbors
    knn_tree = KDTree(reduced_data)  # Build KNN search tree on node features
    @save "knn_tree.jld2" knn_tree
end

dic_knn = Dict()
for sn in keys(node_map)
    source_node=node_map[sn]
    knn_indices = knn(knn_tree, reduced_data[:,source_node], k)
    dic_knn[sn]=knn_indices
end

filtered_edges_knn = DataFrame(x1 = Int[], x2 = Int[])

for row in eachrow(data)
    source_node = row[:x1]
    target_node = row[:x2]
    if target_node in dic_knn[source_node][1]
        push!(filtered_edges_knn, (source_node, target_node))
    end
end =#train_mask, eval_mask, test_mask = random_mask(msc_soft_hot_matrix)



############################ Target preparation
#filt_rel_soft=DataFrame(CSV.File("filt_rel_soft.csv"))
#g_s=SimpleGraph(35220)
#for i in 1:size(filt_rel_soft.col1)[1]
#    add_edge!(g_s, filt_rel_soft.col1[i], filt_rel_soft.col2[i]);
#end


using Random, SparseArrays

# Adjusted function for SparseMatrixCSC with correct concatenation and original sample preservation
function oversample_onehot_sparse(X::SparseMatrixCSC, y::SparseMatrixCSC, label_counts, max_count)
    oversampled_X = [X[:, i] for i in 1:size(X, 2)]  # Start with all original samples
    oversampled_y = [y[:, i] for i in 1:size(y, 2)]  # Include all original labels
    
    for i in 1:size(y, 2)  # Iterate over columns (samples)
        active_labels = findnz(y[:, i])[1]  # Find non-zero entries (active labels) for the sample
        if !isempty(active_labels)
            min_label_count = minimum(label_counts[active_labels])

            # Calculate the oversampling factor
            oversampling_factor = ceil(Int, max_count / min_label_count)

            # Append oversampled columns, but skip appending the original sample (since it's already included)
            for _ in 2:oversampling_factor  # Start from 2 to avoid duplicating the original sample
                push!(oversampled_X, X[:, i])
                push!(oversampled_y, y[:, i])
            end
        end
    end

    # Convert the vector of sparse columns back into sparse matrices
    return sparse(hcat(oversampled_X...)), sparse(hcat(oversampled_y...))
end

X = SparseMatrixCSC{Float32}(Float32.(filtered_msc))
y = SparseMatrixCSC{Float32}(permutedims(Float32.(msc_soft_hot_matrix)))

label_counts = sum(y, dims=2)  # Sum each row (label) to get the frequency
max_count = maximum(label_counts)

X_oversampled, y_oversampled = oversample_onehot_sparse(X, y, label_counts, max_count)

println("Original dataset size: ", size(X, 2))
println("Oversampled dataset size: ", size(X_oversampled, 2))

# Generate the masks with stratified sampling based on multi_hot_matrix
train_mask, eval_mask, test_mask = random_mask(permutedims(X_oversampled))

# Save the masks
serialize("train_mask.jls", train_mask)
serialize("eval_mask.jls", eval_mask)
serialize("test_mask.jls", test_mask)



function extend_graph_with_oversampling(X_oversampled, new_x1, new_x2, desired_num_nodes)
    num_original_nodes = length(unique(vcat(new_x1, new_x2)))  # Number of original nodes
    num_new_nodes = desired_num_nodes - num_original_nodes

    # Create new node indices for the artificial nodes
    if num_new_nodes > 0
        artificial_nodes = collect(num_original_nodes + 1 : desired_num_nodes)
    else
        artificial_nodes = Int[]
    end

    # Initialize new_x1 and new_x2 to accommodate new nodes
    extended_new_x1 = copy(new_x1)
    extended_new_x2 = copy(new_x2)

    # Add new edges for the artificial nodes
    for i in 1:num_new_nodes
        # Connect each artificial node to an existing node
        push!(extended_new_x1, artificial_nodes[i])
        push!(extended_new_x2, rand(1:num_original_nodes))  # Connect to a random original node
        
        # Optionally, connect the new node to another random node or itself
        push!(extended_new_x1, artificial_nodes[i])
        push!(extended_new_x2, artificial_nodes[i])  # Connect to itself or another artificial node
    end

    return extended_new_x1, extended_new_x2, artificial_nodes
end


# Example usage
#X_oversampled = rand(Float32, 63, 214096)  # Oversampled feature matrix with 214096 nodes (features)
#new_x1 = collect(1:63)  # Original source nodes (dummy data)
#new_x2 = collect(1:63)  # Original target nodes (dummy data)

desired_num_nodes = size(X_oversampled)[2]  # The target number of nodes

# Extend the graph with artificial nodes
extended_new_x1, extended_new_x2, artificial_nodes = extend_graph_with_oversampling(X_oversampled, new_x1, new_x2, desired_num_nodes)



# Check the result
println("Total nodes in the graph: ", length(unique(vcat(extended_new_x1, extended_new_x2))))
println("Total edges in the graph: ", length(extended_new_x1))

# Now create the graph with the new nodes and edges
g = GNNGraph(extended_new_x1, extended_new_x2)

g=sample_neighbors(g,all_nodes)

num_nodes = g.num_nodes
source, target = edge_index(g)


g=GNNGraph((source, target))
g=add_nodes(g,1)
### GNN initialization

#g = GNNGraph(new_x1, new_x2)
# Step 2: Define node data (ndata), including features, train_mask, eval_mask, test_mask, and target (P)

ndata = (
    features=X_oversampled, #Float32.(reduced_data),  # The reduced features (e.g., from PCA/SVD)
    train_mask=train_mask,            # Training mask
    eval_mask=eval_mask,              # Evaluation/Validation mask
    test_mask=test_mask,              # Test mask
    target=y_oversampled                       # The target, reduced via SVD (P matrix)
)

g = GNNGraph(g, ndata=ndata)
# Clear the edge data if not needed
  # Empty the edge data store

# Cleaning the memory
new_x1 = nothing
new_x2 = nothing
node_map = nothing
all_nodes = nothing
c = nothing
reduced_data = nothing
msc_soft_hot_matrix = nothing
filtered_edges = nothing
filtered_msc = nothing
train_mask = nothing
eval_mask = nothing
test_mask = nothing

GC.gc()

######################################### MODEL
# Evaluation function for regression tasks (MSE and MAE)


# Arguments structure for the train function
# Rename Args to avoid redefinition error
Base.@kwdef mutable struct TrainingArgs
    η = 1.0f-3             # learning rate
    epochs = 100           # number of epochs
    seed = 17              # set seed > 0 for reproducibility
    usecuda = true         # if true use cuda (if available)
    nhidden = 128          # dimension of hidden features
    batch_size = 4096 #512 #128       # batch size for mini-batch training
    infotime = 10          # report every `infotime` epochs
end

# Function to partition the graph into batches of nodes
function create_batches(g, batch_size)
    # Create batches of node indices for mini-batch training
    node_ids = collect(1:g.num_nodes)
    shuffle!(node_ids)  # Shuffle the node indices to randomize batches
    return [node_ids[i:min(i + batch_size - 1, length(node_ids))]
            for i in 1:batch_size:length(node_ids)]
end

# Function to extract a subgraph and features from a batch of node indices
# Function to extract a subgraph and features from a batch of node indices
# Updated function to extract a subgraph and features from a batch of node indices
# Function to extract a subgraph and features from a batch of node indices
function extract_subgraph(g, node_batch, batch_size)
    num_nodes_in_batch = length(node_batch)  # Actual number of nodes in the batch
    X_batch = Array{Float32}(g.features[:, node_batch])  # Features for the batch

    # Ensure y_batch has the correct number of features (100 in this case)
    y_batch = Array{Float32}(g.target[:, node_batch])  # Adjusted to have 100 features

    # Create a mapping from original node indices to reindexed node IDs
    node_map = Dict(node => idx for (idx, node) in enumerate(node_batch))

    # Get the edges involving only nodes in the batch
    edge_batch = filter(e -> e.src in node_batch && e.dst in node_batch, edges(g))

    # Extract the source and target nodes from the edge_batch
    src_nodes = map(e -> node_map[e.src], edge_batch)
    dst_nodes = map(e -> node_map[e.dst], edge_batch)

    # Combine src_nodes and dst_nodes to ensure unique node indices
    all_nodes = unique(vcat(src_nodes, dst_nodes))

    num_unique_nodes = length(all_nodes)

    # Pad to exactly 128 nodes
    required_size = batch_size 

    if num_unique_nodes < required_size
        # Pad only to reach the required size
        padding_size = required_size - num_unique_nodes
        remaining_nodes = setdiff(1:g.num_nodes, all_nodes)
        padding_nodes = remaining_nodes[1:padding_size]

        all_nodes = vcat(all_nodes, padding_nodes)

        # Add padding to X_batch and y_batch
        padding_X = Array{Float32}(g.features[:, padding_nodes])
        padding_y = Array{Float32}(g.target[:, padding_nodes])  # Ensure the padding has 100 features

        X_batch = hcat(X_batch, padding_X)
        y_batch = hcat(y_batch, padding_y)  # Adjusted to match the target shape

        # Ensure edges are connected for the padding nodes
        padding_edges = [(src, dst) for src in padding_nodes, dst in padding_nodes]
        for (src, dst) in padding_edges
            push!(src_nodes, src)
            push!(dst_nodes, dst)
        end
    end

    # Ensure the batch size is exactly 128
    X_batch = X_batch[:, 1:required_size]
    y_batch = y_batch[:, 1:required_size]  # Adjust to the correct size


    # Create the subgraph using the updated list of nodes and edges
    subgraph = GNNGraph((src_nodes, dst_nodes))

    # Create the subgraph with the correct features and target size
    subgraph = GNNGraph(subgraph, ndata=(
        features=X_batch,  # Assign only up to the required size
        target=y_batch     # Assign only up to the required size
    ))

    return subgraph, X_batch, SparseMatrixCSC(y_batch)
end








function eval_loss_accuracy(X, y, mask, model, g)
    ŷ = model(g, X)

    # Reconstruct the target in binary form
    #reconstructed_target = ŷ[:, mask]' * Q
    best_threshold = 0.5
    binary_predicted_labels = ŷ[:, mask] .> best_threshold
    y_true = y[:, mask]

    # Apply the threshold to y_true for comparison
    y_true_binary = y_true .> best_threshold  # Apply threshold to y_true
    println(size(y_true_binary))
    println(size(binary_predicted_labels))
    # Precision: count true positives (correctly predicted labels) over predicted positives
    precision = sum((binary_predicted_labels .& y_true_binary)) / (sum(binary_predicted_labels) + 1e-6)

    # Recall: count true positives over actual positives
    recall = sum((binary_predicted_labels .& y_true_binary)) / (sum(y_true_binary) + 1e-6)

    # F1-score: harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    println(f1_score)
    # Compute Binary Cross Entropy loss
    l = Flux.Losses.binarycrossentropy(binary_predicted_labels, y_true)

    return round(l, digits=4), round(f1_score, digits=4)
end





# Main training function with updated eval_loss_accuracy
function train(; kws...)
    args = TrainingArgs(; kws...)

    # Set the random seed for reproducibility
    args.seed > 0 && Random.seed!(args.seed)

    # Determine whether to use GPU or CPU
    if args.usecuda && CUDA.functional()
        device = gpu
        args.seed > 0 && CUDA.seed!(args.seed)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # LOAD DATA
    X = g.features
    y = g.target
    num_vec = size(g.target)[1]
    nin, nhidden, nout = size(X, 1), args.nhidden, num_vec

    ## DEFINE MODEL
    model = GNNChain(
        #Dropout(0.5),
        GCNConv(nin => nhidden, relu),
        #Dropout(0.5),
        GCNConv(nhidden => nhidden, relu),
        #Dropout(0.5),
        Dense(nhidden, nout),
        sigmoid
        ) |> device

    # Define the optimizer
    opt = Flux.setup(Adam(args.η), model)

    # LOGGING FUNCTION
    function report(epoch)
        train_loss, train_f1 = eval_loss_accuracy(X, y, g.train_mask, model, g)
        eval_loss, eval_f1 = eval_loss_accuracy(X, y, g.eval_mask, model, g)
        println("Epoch: $epoch   Train Loss: $train_loss  Train F1: $train_f1  Eval Loss: $eval_loss  Eval F1: $eval_f1")
    end

    ## TRAINING LOOP WITH BATCHING
    report(0)  # Initial evaluation before training

    for epoch in 1:args.epochs
        println("Epoch: $epoch") 
        batches = create_batches(g, args.batch_size)

        for batch in batches
            # Extract the subgraph, features, and targets for the batch
            subgraph, X_batch, y_batch = extract_subgraph(g, batch, args.batch_size)

            # Move the data to the appropriate device (GPU/CPU)
            X_batch, y_batch = X_batch |> device, y_batch |> device

            # Perform forward and backward pass for the batch
            grad = Flux.gradient(model) do model
                ŷ = model(subgraph, X_batch)
                # Use binary cross-entropy instead of MSE
                Flux.Losses.binarycrossentropy(ŷ, y_batch)
            end

            # Update the model parameters
            Flux.update!(opt, model, grad[1])
        end

        # Report every `infotime` epochs
        epoch % args.infotime == 0 && report(epoch)
    end

    # Save the model after training
    BSON.@save "model.bson" model opt

    # Final evaluation on the test set
    test_loss, test_f1 = eval_loss_accuracy(X, y, g.test_mask, model, g, Q)
    println("Final Test Loss: $test_loss  Final Test F1: $test_f1")
end


train()