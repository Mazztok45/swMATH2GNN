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

df= DataFrame(Arrow.Table("GNN_Julia/df_arrow"))



#df.msc_codes_2 = map(y-> unique(map(x -> SubString(x,1:2),y)), df.msc_codes)


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
### Unique set of parent MSC codes
maj_cat = filter!(!isempty,unique(reduce(vcat, df.msc_codes_2)))


###Encoding the nodes
#enc=labelenc(cat_nodes)
#cat_nodes_enc=map(x -> enc.invlabel[x],cat_nodes)
#maj_cat_enc=map(x -> enc.invlabel[x],map(x -> x[1:2],cat_nodes))

#CSV.write("msc_edges.csv", DataFrame(col1=maj_cat_enc,col2=cat_nodes_enc))

####new strat
enc=labelenc(maj_cat)
### Get the encoded MSC node features
enc_node_features=[map(x -> enc.invlabel[x[1:2]], y) for y in df.msc_codes_2]

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
function msc_encoding()
    #return DataFrame(Arrow.Table("GNN_Julia/msc_arrow/arrow"))
    return deserialize("dense_one_hot.jls")
end



unique_paper_ids = Set(unique(df.paper_id))
#= 
function paper_edges()
    return DataFrame(Arrow.Table("GNN_Julia/papers_edges_arrow/papers_edges_arrow"))
end


filtered_data = filter(row -> row.paper_id in unique_paper_ids, unique(df[!, [:paper_id, :software]]))
 =#
grouped_data_by_paper_id = combine(groupby(df, :paper_id), :software => x -> collect(x) => :software_array, :msc_codes => x -> collect(x) => :msc_array)


merged_msc_column = [reduce(vcat, vec(pair.first)) for pair in grouped_msc_by_paper_id.msc_codes_function]
grouped_data_by_paper_id.merged_msc_codes = merged_msc_column
# Extract the software arrays from the first element of each pair
grouped_data_by_paper_id.merged_software = [pair.first for pair in grouped_by_paper_id.software_function]

software_arrays = grouped_data_by_paper_id.merged_software


grouped_data_by_paper_id.merged_msc_codes_2 = map(y-> unique(map(x -> SubString(x,1:2),y)), grouped_data_by_paper_id.merged_msc_codes)

grouped_data_by_paper_id.merged_msc_codes_2 = join.(grouped_data_by_paper_id.merged_msc_codes_2, ",")
grouped_data_by_paper_id.merged_software = join.(grouped_data_by_paper_id.merged_software, ",")

selected_columns = grouped_data_by_paper_id[:, [:paper_id, :merged_msc_codes_2, :merged_software]]

# Export to CSV
CSV.write("grouped_data_by_paper_id.csv", selected_columns )


########### graph on software mention distribution amongs papers
using Plots
counts = Dict{String, Int}()

# Iterate over the nested arrays and count each string
for arr in software_arrays
    for value in arr
        counts[value] = get(counts, value, 0) + 1
    end
end


# Prepare data for plotting
string_counts = collect(values(counts))  # List of counts only

# Define y-tick positions (from 0 to 28000 with a step of 5000)
ytick_positions = 0:5000:28000

# Plotting the histogram with custom y-ticks
hist_plot=histogram(
    string_counts, 
    bins=10, 
    title="Histogram of Software Occurrences", 
    xlabel="Software Occurrences Count", 
    ylabel="Frequency", 
    yticks=(ytick_positions, string.(ytick_positions)),# An example of semi-supervised node classification
    label=false   # Use the defined y-ticks with integer labels
)
# Save the plot to a file (e.g., PNG)
savefig(hist_plot, "software_mentions_distribution.png")
########### 