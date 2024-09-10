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

u_soft = size(unique(vcat(filt_rel_soft.col1,filt_rel_soft.col2)))[1]  ## 32807

maj_cat_enc=DataFrame(CSV.File("msc_edges.csv")).col1
cat_nodes_enc=DataFrame(CSV.File("msc_edges.csv")).col2


 

soft_enc = map(x->parse(Int64,x),split(read("soft_enc.csv",String), '\n'))

num_nodes = Dict(:msc => 5571, :software => 35220, :paper => 332669)#size(unique(vcat(filt_rel_soft.col1,filt_rel_soft.col2)))[1]

rand_paper_id = [1:332669;]
msc_to_soft = DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow_msc_to_soft"))


data = ((:software,:relates_to, :software)=> (filt_rel_soft.col1, filt_rel_soft.col2),
(:msc,:parent_cat,:msc)=>(maj_cat_enc, cat_nodes_enc),
(:msc,:parent_cat,:software)=>(Vector(msc_to_soft.x1), Vector(msc_to_soft.x2)),
(:paper,:p2s,:software)=>(rand_paper_id,soft_enc))


soft_mat = permutedims(unique(soft_enc) .== permutedims(soft_enc))

### x dim is (332669, 5571) and y dim (332669, 35220)
msc_features=msc_encoding()

msc_features=permutedims(msc_features)
soft_enc=permutedims(soft_enc)

ndata=Dict(:paper => (x = msc_features,y=soft_enc))
G = GNNHeteroGraph(data; num_nodes, ndata) 