if basename(pwd()) != "swMATH2GNN"
    cd("..")
end 
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
using Serialization
using CSV

if !isfile("GNN_Julia/df_arrow")
    ########### Section to run if df_arrow does not exist ###########
    articles_list_dict = extract_articles_metadata()
    node_features = [dic[:msc] for dic in articles_list_dict]
    art_soft = [dic[:software] for dic in articles_list_dict]
    paper_id_soft = [dic[:id] for dic in articles_list_dict]
    titles = [dic[:title] for dic in articles_list_dict]

    refs_soft = [dic[:ref_ids] for dic in articles_list_dict]
    df = unique(sort!(DataFrame(paper_id=paper_id_soft, msc_codes=node_features, title=titles, software=art_soft)))
    Arrow.write("GNN_Julia/df_arrow", df)
end


########## 
df= DataFrame(Arrow.Table("GNN_Julia/df_arrow"))

#= 

map(x -> split(joinx,";"), split(read("msc.txt",String), '\n')) =#
#= 


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
 =#
### Unique set of MSC codes
#= cat_nodes = filter!(!isempty,unique(reduce(vcat, new_nodes_features))) =#


###Encoding the nodes
#enc=labelenc(cat_nodes)
#cat_nodes_enc=map(x -> enc.invlabel[x],cat_nodes)
#maj_cat_enc=map(x -> enc.invlabel[x],map(x -> x[1:2],cat_nodes))

#CSV.write("msc_edges.csv", DataFrame(col1=maj_cat_enc,col2=cat_nodes_enc))


####### BEGIN SECTION NOT USEFUL ANYMORE #######
df.msc_codes_2 = map(y-> unique(map(x -> SubString(x,1:2),y)), df.msc_codes)
maj_cat = filter!(!isempty,unique(reduce(vcat, df.msc_codes_2)))
enc=labelenc(maj_cat)
### Get the encoded MSC node features
enc_node_features=[map(x -> enc.invlabel[x[1:2]], y) for y in df.msc_codes_2]

df_2 = DataFrame(msc=enc_node_features)
ux = unique(reduce(vcat, df_2.msc))
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
####### END SECTION NOT USEFUL ANYMORE #######

#= ##### Articles MSC codes data processing
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
 =#
### Unique set of MSC codes
#= cat_nodes = filter!(!isempty,unique(reduce(vcat, new_nodes_features))) =#
### Uni
#= dense_one_hot=deserialize("dense_one_hot.jls") = =#


###### SOME FURTHER STEPS AT PREPARING THE DATA BEFORE IMPLEMENTING THE MODELS
#function msc_encoding()
    #return DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow"))
 #   return deserialize("dense_one_hot.jls")
#end



#unique_paper_ids = Set(unique(df.paper_id))
#= 
function paper_edges()
    return DataFrame(Arrow.Table("GNN_Julia/papers_edges_arrow/papers_edges_arrow"))
end


filtered_data = filter(row -> row.paper_id in unique_paper_ids, unique(df[!, [:paper_id, :software]]))
 =#

 if !isfile("grouped_data_by_paper_id.csv")
    grouped_data_by_paper_id = combine(groupby(df, :paper_id), :software => x -> collect(x) => :software_array, :msc_codes => x -> collect(x) => :msc_array, :title)


    merged_msc_column = [reduce(vcat, vec(pair.first)) for pair in grouped_data_by_paper_id.msc_codes_function]
    grouped_data_by_paper_id.merged_msc_codes = merged_msc_column
    # Extract the software arrays from the first element of each pair


    grouped_data_by_paper_id.merged_msc_codes_2 = map(y-> unique(map(x -> SubString(x,1:2),y)), grouped_data_by_paper_id.merged_msc_codes)
    grouped_data_by_paper_id.merged_msc_codes_2 = join.(grouped_data_by_paper_id.merged_msc_codes_2, ",")

    grouped_data_by_paper_id.merged_software = join.([pair.first for pair in grouped_data_by_paper_id.software_function], ",")

    selected_columns = grouped_data_by_paper_id[:, [:paper_id, :merged_msc_codes_2, :merged_software, :title]]

    # Export to CSV
    CSV.write("grouped_data_by_paper_id.csv", selected_columns)
 end
selected_columns = DataFrame(CSV.File("grouped_data_by_paper_id.csv"))
software_arrays = map(x -> parse.(Int, split(x, ",")), selected_columns.merged_software)

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


serialize("multi_hot_matrix.jls", multi_hot_matrix)

##

function msc_encoding()
    #return DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow"))
    return deserialize("dense_one_hot.jls")
end

selected_paper_id = Set(unique(grouped_by_paper_id.paper_id))

sd = setdiff(unique_paper_ids, selected_paper_id)

vec_u = collect(unique_paper_ids)


l_ind = [i for (i, j) in enumerate(vec_u) if !(j in sd)]

filtered_msc = permutedims(msc_encoding()[l_ind, :])

serialize("filtered_msc.jls", filtered_msc)



X = SparseMatrixCSC{Float32}(Float32.(filtered_msc))
y = SparseMatrixCSC{Float32}(permutedims(Float32.(msc_soft_hot_matrix)))


### Export the Edges between related software from the software graph
software_df = generate_software_dataframe()

related_soft_edgelist = sortslices(hcat(software_df.id, map(x -> x[:id], software_df.related_software)),dims=1,by=x->(x[1],x[2]),rev=false)

unique_software_in_articles = unique(reduce(vcat, software_arrays))

software_edges = reduce(hcat,filter(x -> x[1] in unique_software_in_articles && x[2] in unique_software_in_articles, eachrow(related_soft_edgelist)))'




CSV.write("software_graph2.edgelist", Tables.table(software_edges ))
