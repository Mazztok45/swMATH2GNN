using CSV
using DataFrames
using GraphNeuralNetworks
using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using Random
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
import GraphNeuralNetworks: GNNHeteroGraph, GNNGraphs
using OneHotArrays
using MLLabelUtils
using StatsBase
using Arrow
##### GRAPH PREP

#ds = Parquet2.Dataset("GNN_Julia/msc_parquet/parquet")
#df = DataFrame(ds; copycols=false)
function msc_encoding()
    return Matrix(DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow")))
end


## Two columns DataFrame of related each other software
filt_rel_soft=DataFrame(CSV.File("filt_rel_soft.csv"))

#u_soft = size(unique(vcat(filt_rel_soft.col1,filt_rel_soft.col2)))[1]  ## 32807

maj_cat_enc=DataFrame(CSV.File("msc_edges.csv")).col1
cat_nodes_enc=DataFrame(CSV.File("msc_edges.csv")).col2


 

soft_enc = map(x->parse(Int64,x),split(read("soft_enc.csv",String), '\n'))

num_nodes = Dict(:software => 35220, :paper => 332669,:msc => 5571)#size(unique(vcat(filt_rel_soft.col1,filt_rel_soft.col2)))[1]

rand_paper_id = [1:332669;]
msc_to_soft = DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow_msc_to_soft"))


data = ((:software,:relates_to, :software)=> (filt_rel_soft.col1, filt_rel_soft.col2),
(:msc,:parent_cat,:msc)=>(maj_cat_enc, cat_nodes_enc),
(:msc,:parent_cat,:software)=>(Vector(msc_to_soft.x1), Vector(msc_to_soft.x2)),
(:paper,:p2s,:software)=>(rand_paper_id,soft_enc))


#soft_mat = permutedims(unique(soft_enc) .== permutedims(soft_enc))

### x dim is (332669, 5571) and y dim (332669, 35220)
msc_features_np=Matrix{Float32}(msc_encoding())
msc_features=permutedims(Matrix{Float32}(msc_encoding()))
#soft_enc=permutedims(soft_enc)

ndata=Dict(:paper => (x =msc_features_np,y=soft_enc))
G = GNNHeteroGraph(data; num_nodes, ndata) 

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

    #if args.usecuda && CUDA.functional()
        #device = gpu
        #args.seed > 0 && CUDA.seed!(args.seed)
        #@info "Training on GPU"
    #else
    device = cpu
    @info "Training on CPU"
    #end

    # LOAD DATA
    #dataset = Cora()
    #classes2=dataset.metadata["classes"]
    #g = mldataset2gnngraph(dataset)# |> device
    X = G[:paper].x
    println(typeof(X))
    classes = sort(unique(G[:paper].y))
    y=onehotbatch(G[:paper].y |> cpu, classes) # remove when https://github.com/FluxML/Flux.jl/pull/1959 tagged
    println(typeof(y))
    #y2=onehotbatch(g.targets |> cpu, classes2)
   
    train_mask, test_mask = random_mask()
    
    ytrain = y[:, train_mask]

    nin, nhidden, nout = size(X, 1), args.nhidden, length(classes)
    println(nin)
    println(nhidden)
    ## DEFINE MODEL
    #model = GNNChain(GCNConv(nin => nhidden; relu),
    #        GCNConv(nhidden => nhidden; relu),
    #                 Dense(nhidden, nout)) |> device

    layer = HeteroGraphConv(
            (:paper, :p2s, :software) => GraphConv(5571 => 128, relu)
                     );
    
    nx=permutedims(G.ndata[:paper].x)
    y=layer(G,(paper=nx,software=G.ndata[:paper].y))

                        
    model=GNNChain(layer)
    opt = Flux.setup(Adam(args.η), model)

    display(G) #display(g)

    ## LOGGING FUNCTION
    function report(epoch)
        train = eval_loss_accuracy(X, y, train_mask, model, G)
        test = eval_loss_accuracy(X, y, test_mask, model, G)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end

    ## TRAINING
    report(0)
    for epoch in 1:(args.epochs)
        grad = Flux.gradient(model) do model
            ŷ = model(g, X)
            logitcrossentropy(ŷ[:, train_mask], ytrain)
        end

        Flux.update!(opt, model, grad[1])

        epoch % args.infotime == 0 && report(epoch)
    end
end

train()