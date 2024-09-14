# An example of semi-supervised node classification
using Flux
using Flux: onecold, onehotbatch
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

function msc_encoding()
    return DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow"))
end

#msc_features_np=Matrix{Float32}(msc_encoding())


pi_int = parse.(Int, split(read("msc_paper_id.txt",String), '\n'))

function paper_edges()
    return DataFrame(Arrow.Table("GNN_Julia/papers_edges_arrow/papers_edges_arrow"))
end

refs_df=paper_edges()

hcat_df=hcat(pi_int, msc_encoding())

unique_paper_ids = Set(unique(hcat_df.x1))


filtered_data = filter(row -> row.paper_id in unique_paper_ids, unique(refs_df[!,[:paper_id,:software]]))

grouped_by_paper_id = combine(groupby(filtered_data, :paper_id), :software => collect => :software_array)


# Extract the software arrays from the first element of each pair
software_arrays = [pair.first for pair in grouped_by_paper_id.software_function]



### CODE GOOD UNTIL THERE !!

num_nodes = Dict(:paper => 146346) #size(unique(vcat(filt_rel_soft.col1,filt_rel_soft.col2)))[1]


data = (
(:paper,:cited_by,:paper)=>(Vector(paper_edges_df.ref_id), Vector(paper_edges_df.paper_id))
)


grouped_data = Dict{Int, Vector{Int}}()
refs_df.paper_id=Int.(refs_df.paper_id)
refs_df.software=map(x -> parse(Int, x), refs_df.software)


for row in eachrow(refs_df)
    key = row[:paper_id]
    value = row[:software]
    
    if haskey(grouped_data, key)
        push!(grouped_data[key], value)
    else
        grouped_data[key] = [value]
    end
end
soft_mat= Vector{Vector}()
for key in keys(grouped_data)
    push!(soft_mat,grouped_data[key])
end








function random_mask()
    at=0.7
    n = size(rand_paper_id,1)
    vec=shuffle(rand_paper_id)
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

ndata=Dict(:paper => (x =msc_features_np,y=soft_enc))
G = GNNHeteroGraph(data; num_nodes, features, train_mask,test_mask) 




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
    dataset = Cora()
    classes = dataset.metadata["classes"]
    g = mldataset2gnngraph(dataset) |> device
    X = g.features
    y = onehotbatch(g.targets |> cpu, classes) |> device # remove when https://github.com/FluxML/Flux.jl/pull/1959 tagged
    ytrain = y[:, g.train_mask]

    nin, nhidden, nout = size(X, 1), args.nhidden, length(classes)

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