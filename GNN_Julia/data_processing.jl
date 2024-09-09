using CSV
using DataFrames
#using GraphNeuralNetworks
#using Flux
#using TextAnalysis
#using MultivariateStats
using SparseArrays
#using StructTypes
#using Graphs
#using Combinatorics
#include("hetero_data.jl")
include("../extracting_data/extract_software_metadata.jl")
include("../extracting_data/extract_article_metadata.jl")
#using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe
import .DataReaderszbMATH: extract_articles_metadata
#import MultivariateStats: fit, KernelPCA
#import GraphNeuralNetworks: GNNHeteroGraph, GNNGraphs
#using OneHotArrays
using MLLabelUtils
using StatsBase
using Arrow

########### FUNCTIONS THAT MUST BE KEPT ###########
#articles_list_dict = extract_articles_metadata()

#node_features = [dic[:msc] for dic in articles_list_dict]
#art_soft = [dic[:software] for dic in articles_list_dict]
###


### Function to write articles MSC
#msc_vec = [join(elem,";") for elem in node_features]
#msc_str = join(msc_vec,'\n')
#open("msc.txt", "w") do file
#    write(file, msc_str)
#end
###

### Function to write articles articles software labels
#art_soft_str = join(art_soft,'\n')
#open("art_soft.txt", "w") do file
#    write(file, art_soft_str)
#end
########### END ###########


##### Articles MSC codes data processing
### Function to read articles msc
node_features = map(x -> split(x,";"), split(read("msc.txt",String), '\n'))
#### Remove empty MSC codes
filt_empt_nf =[filter!(x -> !isempty(x), y) for y in node_features if isempty(y)==false]
### Extending node_features with parent MSC codes
new_nodes_features= Vector{Vector}()
maj_cat_vector = Vector{Vector}()
for l in filt_empt_nf
    maj_codes=unique([str_msc[1:2] for str_msc in l])
    push!(new_nodes_features, vcat(l,maj_codes))
    push!(maj_cat_vector, maj_codes)
end

### Unique set of MSC codes
cat_nodes = filter!(!isempty,unique(reduce(vcat, new_nodes_features)))
### Unique set of parent MSC codes
maj_cat = filter!(!isempty,unique(reduce(vcat, maj_cat_vector)))


###Encoding the nodes
enc=labelenc(cat_nodes)
cat_nodes_enc=map(x -> enc.invlabel[x],cat_nodes)
maj_cat_enc=map(x -> enc.invlabel[x],map(x -> x[1:2],cat_nodes))
DataFrame(col1=maj_cat_enc,col2=cat_nodes_enc)

CSV.write("msc_edges.csv", DataFrame(col1=maj_cat_enc,col2=cat_nodes_enc))


### Get the encoded MSC node features
enc_node_features=[map(x -> enc.invlabel[x], y) for y in new_nodes_features]

df = DataFrame(msc=enc_node_features)
ux = unique(reduce(vcat, df.msc))
sort!(ux)
value_to_index = Dict(value => i for (i, value) in enumerate(ux))

n_rows = size(df)[1]
n_columns = length(ux)

one_hot_matrix = spzeros(Int, n_rows, n_columns)

# Populate the matrix with 1s where needed
for (i, vector) in enumerate(enc_node_features)
    for value in vector
        column_index = value_to_index[value]
        one_hot_matrix[i, column_index] = 1
    end
end

dense_one_hot = Matrix(one_hot_matrix)

df_enc = DataFrame(dense_one_hot, Symbol.(string.(ux)))

#### One hot array
#df_enc = transform(df, :msc .=> [ByRow(v -> x in v) for x in ux] .=> Symbol.(:msc_, ux))
### Changing the types
#for x in names(df_enc)[2:5509]
#    df_enc[!,x] = convert.(Int,df_enc[!,x])
#end

#CSV.write("msc_encoding_mat.csv", df_enc[:,2:5509])
#file = tempname()*".parquet"
#write_parquet(file, df_enc[:,2:5509])

#Parquet2.writefile("GNN_Julia/msc_parquet/parquet", df_enc[:,2:5509])
Arrow.write("GNN_Julia/msc_arrow/arrow", df_enc)



##### Software data processing
## Reading data
software_df  = generate_software_dataframe()
art_soft = split(read("art_soft.txt",String), '\n')

### Encoding software
enc_soft=labelenc(art_soft)
soft_enc=map(x -> enc_soft.invlabel[x],art_soft)


soft_enc_str = join(soft_enc,'\n')
open("soft_enc.csv", "w") do file
    write(file, soft_enc_str)
end



u_soft = map(x->parse(Int64,x),unique(art_soft))

related_soft = [collect(row)[1][2] for row in software_df.related_software]

df_rel_soft=DataFrame(col1=software_df.id, col2=related_soft)
filt_rel_soft = filter(row -> row.col1 in u_soft && row.col2 in u_soft, df_rel_soft)

CSV.write("filt_rel_soft.csv", filt_rel_soft )

