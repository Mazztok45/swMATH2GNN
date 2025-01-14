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
using Serializationart_soft = [dic[:software] for dic in articles_list_dict]

########### FUNCTIONS THAT MUST BE KEPT ###########
articles_list_dict = extract_articles_metadata()
software_df = generate_software_dataframe()

node_features = [dic[:msc] for dic in articles_list_dict]
art_soft = [dic[:software] for dic in articles_list_dict]
paper_id_soft = [dic[:id] for dic in articles_list_dict]


## prepare references to only keep articles in references which mention software
refs_soft = [dic[:ref_ids] for dic in articles_list_dict]


df = unique(sort!(DataFrame(paper_id=paper_id_soft, msc_codes=node_features,software=art_soft)))
Arrow.write("GNN_Julia/df_arrow", df)


join(apply(x -> x[1:2],df.msc_codes[1]),";")


map(y-> unique(map(x -> SubString(x,1:2),y)), df.msc_codes)




map(x -> split(joinx,";"), split(read("msc.txt",String), '\n'))
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





#= 

#msc_vec = [join(elem,";") for elem in grouped_nodes.concatenated_list]
msc_str = join(grouped_nodes.concatenated_list,'\n')
open("msc.txt", "w") do file
    write(file, msc_str)
end


msc_str = join(grouped_nodes.col1,'\n')
open("msc_paper_id.txt", "w") do file
    write(file, msc_str)
end
###
 =#
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
#enc=labelenc(cat_nodes)
#cat_nodes_enc=map(x -> enc.invlabel[x],cat_nodes)
#maj_cat_enc=map(x -> enc.invlabel[x],map(x -> x[1:2],cat_nodes))

#CSV.write("msc_edges.csv", DataFrame(col1=maj_cat_enc,col2=cat_nodes_enc))

####new strat
enc=labelenc(maj_cat)
### Get the encoded MSC node features
enc_node_features=[map(x -> enc.invlabel[x[1:2]], y) for y in new_nodes_features]

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

dense_one_hot = BitArray(Matrix(one_hot_matrix))

serialize("dense_one_hot.jls",dense_one_hot)
dense_one_hot=deserialize("dense_one_hot.jls")

#df_enc = DataFrame(dense_one_hot, Symbol.(string.(ux)))

#Arrow.write("GNN_Julia/msc_arrow/arrow", df_enc)



##### Software data processing
## Reading data
#software_df  = generate_software_dataframe()
#art_soft = split(read("art_soft.txt",String), '\n')

### Encoding software
#enc_soft=labelenc(art_soft)
#soft_enc=map(x -> enc_soft.invlabel[x],art_soft)


#####CHATGPT - MSC to software mapping
#function collect_msc_to_soft(enc_node_features, soft_enc)
    # Estimate the size for the edge list if possible
#    total_edges = sum(length(l_msc) for l_msc in enc_node_features)  # Total expected number of edges

    # Preallocate storage for edges (MSC -> Software pairs)
#    msc_to_soft = Vector{Vector{Int}}(undef, total_edges)

    # Track index
#    edge_index = 1

    # Build the edge list
#    for (i, soft) in enumerate(soft_enc)
#        l_msc = enc_node_features[i]  # Get MSC codes for this software
#        for msc in l_msc
#            msc_to_soft[edge_index] = [msc, soft]
#            edge_index += 1
#        end
#    end

#    return msc_to_soft
#end

# Example usage
# enc_node_features: Vector of Vectors where each inner vector holds MSC codes for a paper/software
# soft_enc: Vector of software labels corresponding to each paper/software
#msc_to_soft = collect_msc_to_soft(enc_node_features, soft_enc)
#Arrow.write("GNN_Julia/msc_arrow/arrow_msc_to_soft", DataFrame(mapreduce(permutedims, vcat, msc_to_soft), :auto))
#####CHATGPT END

#soft_enc_str = join(soft_enc,'\n')
#open("soft_enc.csv", "w") do file
#    write(file, soft_enc_str)
#end



#u_soft = map(x->parse(Int64,x),unique(art_soft))

#related_soft = [collect(row)[1][2] for row in software_df.related_software]

#df_rel_soft=DataFrame(col1=software_df.id, col2=related_soft)
#filt_rel_soft = filter(row -> row.col1 in u_soft && row.col2 in u_soft, df_rel_soft)
#filt_rel_soft.col1= string.(filt_rel_soft.col1)
#filt_rel_soft.col2= string.(filt_rel_soft.col2)
#filt_rel_soft = DataFrame(col1=map(x -> enc_soft.invlabel[x],filt_rel_soft.col1), col2=map(x -> enc_soft.invlabel[x],filt_rel_soft.col2))
#CSV.write("filt_rel_soft.csv", filt_rel_soft )

