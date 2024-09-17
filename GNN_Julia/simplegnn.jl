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
#using MultivariateStats
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




function msc_encoding()
    #return DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow"))
    return deserialize("dense_one_hot.jls")
end

#msc_features_np=Matrix{Float32}(msc_encoding())


pi_int = parse.(Int, split(read("msc_paper_id.txt",String), '\n'))
unique_paper_ids = Set(unique(pi_int))

function paper_edges()
    return DataFrame(Arrow.Table("GNN_Julia/papers_edges_arrow/papers_edges_arrow"))
end




filtered_data = filter(row -> row.paper_id in unique_paper_ids, unique(paper_edges()[!,[:paper_id,:software]]))

grouped_by_paper_id = combine(groupby(filtered_data, :paper_id), :software => x -> collect(x) => :software_array)


# Extract the software arrays from the first element of each pair
software_arrays = [pair.first for pair in grouped_by_paper_id.software_function]

# Cleaning the memory
pi_int=nothing
unique_paper_ids =nothing
filtered_data =nothing
grouped_by_paper_id=nothing

GC.gc()

# Collect all unique software labels
all_software_labels = unique(vcat(software_arrays...))

# Map each software label to a unique index
label_to_index = Dict(label => i for (i, label) in enumerate(all_software_labels))


# Number of unique software labels
num_labels = length(all_software_labels)

# Initialize an empty array to store multi-hot vectors
#multi_hot_encoded = Vector{Vector{Int}}()

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
num_labels = length(all_software_labels)
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

multi_hot_matrix=permutedims(multi_hot_matrix)

serialize("mutli_hot_matrix.jls",multi_hot_matrix)
#Arrow.write("GNN_Julia/multi_hot_encoded.csv", DataFrame(Matrix{Bool}(permutedims(multi_hot_matrix)), :auto))
# Display one of the multi-hot encoded vectors for checking
#println(multi_hot_encoded[1])


selected_paper_id=Set(unique(grouped_by_paper_id.paper_id))

sd = setdiff(unique_paper_ids,selected_paper_id)

vec_u = collect(unique_paper_ids)


l_ind = [i for (i,j) in enumerate(vec_u) if !(j in sd)]

filtered_msc =  msc_encoding()[l_ind,:]

serialize("filtered_msc.jls",filtered_msc)

#Arrow.write("GNN_Julia/filtered_msc",filtered_msc)





### ADAPT THE EDGES
refs_df=paper_edges()

filtered_edges = filter(row -> (row.paper_id in unique_paper_ids  && !(row.paper_id  in sd)), unique(paper_edges()[!,[:paper_id,:ref_id]]))

# Assuming l is a list of ref_ids
l = unique(filtered_edges.paper_id)
# Convert l to a Set for efficient membership checking
l_set = Set(l)
# Create refs_id2 by iterating through each row
filtered_edges.refs_id2 = [row.ref_id in l_set ? row.ref_id : row.paper_id for row in eachrow(filtered_edges)]



Arrow.write("GNN_Julia/filtered_edges", filtered_edges)


######## SPLITING MASKS



filtered_edges= DataFrame(Arrow.Table(Arrow.read("GNN_Julia/filtered_edges")))


function random_mask()
    at=0.7
    n = size(grouped_by_paper_id.paper_id,1)
    vec=shuffle(grouped_by_paper_id.paper_id)
    train_idx = view(vec, 1:floor(Int, at*n))
    #test_idx = view(vec, (floor(Int, at*n)+1):n)
    n_train=size(train_idx,1)
    #n_test=size(test_idx,1)
    bool_train = vcat(collect((1 for n=1:n_train)),collect((0 for n=1:(n-n_train))))
    bool_test = vcat(collect((0 for n=1:n_train)),collect((1 for n=1:(n-n_train))))
    dftr = DataFrame(col1=vec,col2=bool_train)
    dfte = DataFrame(col1=vec,col2=bool_test)
    train_mask = BitVector(sort!(dftr,[:col1])[:,:col2])
    test_mask = BitVector(sort!(dfte,[:col1])[:,:col2])
    return train_mask, test_mask
end

train_mask, test_mask = random_mask()
serialize("train_mask.jls",train_mask)
serialize("test_mask.jls",test_mask)
#Arrow.write("GNN_Julia/random_mask", DataFrame(train_mask=train_mask,test_mask=test_mask))




#### MODEL PART



filtered_edges= DataFrame(Arrow.Table(Arrow.read("GNN_Julia/filtered_edges")))
multi_hot_matrix= deserialize("mutli_hot_matrix.jls")
filtered_msc= deserialize("filtered_msc.jls")
train_mask = deserialize("train_mask.jls")
test_mask = deserialize("test_mask.jls")

#DataFrame(Arrow.Table(Arrow.read("GNN_Julia/random_mask"))).train_mask, DataFrame(Arrow.Table(Arrow.read("GNN_Julia/random_mask"))).test_mask




num_nodes = Dict(:paper => 146346)

#data = ((:paper,:cited_by,:paper)=>(Vector(filtered_edges.refs_id2), Vector(filtered_edges.paper_id)))

data = unique(DataFrame(hcat(Vector(filtered_edges.refs_id2), Vector(filtered_edges.paper_id)), :auto))


# Combine all node IDs and find unique ones
all_nodes = unique(vcat(data.x1, data.x2))

# Create a dictionary to map old node IDs to new sequential IDs
node_map = Dict(node => i for (i, node) in enumerate(all_nodes))

# Remap the node IDs in your data
new_x1 = [node_map[node] for node in data.x1]
new_x2 = [node_map[node] for node in data.x2]

# Create a new GNNGraph with remapped node IDs
g = GNNGraph(new_x1, new_x2)


ndata=(features = permutedims(filtered_msc), targets=permutedims(multi_hot_matrix), train_mask=train_mask, test_mask=test_mask)



g = GNNGraph(g, ndata=ndata)


function eval_loss_accuracy(X, y, mask, model, g)
    ŷ = model(g, X)
    l = logitcrossentropy(ŷ[:, mask], y[:, mask])
    acc = mean(onecold(ŷ[:, mask]) .== onecold(y[:, mask]))
    return (loss = round(l, digits = 4), acc = round(acc * 100, digits = 2))
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1.0f-3             # learning rate
    epochs = 100          # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = true      # if true use cuda (if available)
    nhidden = 128        # dimension of hidden features
    infotime = 10      # report every `infotime` epochs
end

function train(; kws...)
    args = Args(; kws...)

    args.seed > 0 && Random.seed!(args.seed)

    if args.usecuda && CUDA.functional()
        device = gpu
        args.seed > 0 && CUDA.seed!(args.seed)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # LOAD DATA
    #dataset = Cora()
    #classes = dataset.metadata["classes"]
    #g = mldataset2gnngraph(dataset) |> device
    X = g.features
    y = g.targets #onehotbatch(g.targets |> cpu, classes) |> device # remove when https://github.com/FluxML/Flux.jl/pull/1959 tagged
    ytrain = y[:, g.train_mask]

    nin, nhidden, nout = size(X, 1), args.nhidden, length(values(label_to_index))

    ## DEFINE MODEL
    model = GNNChain(GCNConv(nin => nhidden, relu),
                     GCNConv(nhidden => nhidden, relu),
                     Dense(nhidden, nout)) |> device

    opt = Flux.setup(Adam(args.η), model)

    display(g)

    ## LOGGING FUNCTION
    function report(epoch)
        train = eval_loss_accuracy(X, y, g.train_mask, model, g)
        test = eval_loss_accuracy(X, y, g.test_mask, model, g)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end

    ## TRAINING
    report(0)
    for epoch in 1:(args.epochs)
        grad = Flux.gradient(model) do model
            ŷ = model(g, X)
            logitcrossentropy(ŷ[:, g.train_mask], ytrain)
        end

        Flux.update!(opt, model, grad[1])

        epoch % args.infotime == 0 && report(epoch)
    end
end

train()

