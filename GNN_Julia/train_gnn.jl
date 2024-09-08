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

data = ((:software,:relates_to, :software)=> (filt_rel_soft.col1, filt_rel_soft.col2),
(:msc,:parent_cat,:msc)=>(maj_cat_enc, cat_nodes_enc))

num_nodes = Dict(:msc => 5571, :software => size(unique(vcat(filt_rel_soft.col1,filt_rel_soft.col2)))[1]
)

soft_enc = map(x->parse(Int64,x),split(read("soft_enc.csv",String), '\n'))


soft_mat = permutedims(unique(soft_enc) .== permutedims(soft_enc))

ndata=Dict(:msc => (x = msc_encoding()),:software =>(y=soft_mat))

G = GNNHeteroGraph(data; num_nodes, ndata)