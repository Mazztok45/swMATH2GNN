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
using Distributed
using HTTP
using JSON3
using URIs
using Logging
using FilePathsBase

if !isfile("GNN_Julia/df_arrow")
    ########### Section to run if df_arrow does not exist ###########
    articles_list_dict = extract_articles_metadata()

    if !isfile("GNN_Julia/node_features")
        println("Preparing node_features")
        node_features = [dic[:msc] for dic in articles_list_dict]
        Arrow.write("GNN_Julia/node_features", Tables.table(node_features, header=["node_features"]))
        node_features=missing
        GC.gc()
    end

    if !isfile("GNN_Julia/art_soft")
        println("Preparing art_soft")
        art_soft = [dic[:software] for dic in articles_list_dict]
        Arrow.write("GNN_Julia/art_soft", Tables.table(art_soft, header=["software"]))
        art_soft=missing
        GC.gc()
    end

    if !isfile("GNN_Julia/paper_id_soft")
        println("Preparing paper_id_soft")
        paper_id_soft = [dic[:id] for dic in articles_list_dict]
        Arrow.write("GNN_Julia/paper_id_soft", Tables.table(paper_id_soft, header=["paper_id_soft"]))
        paper_id_soft=missing
        GC.gc()
    end

    if !isfile("GNN_Julia/titles")
        println("Preparing titles")
        titles = [dic[:title] for dic in articles_list_dict]
        Arrow.write("GNN_Julia/titles", Tables.table(titles, header=["title"]))
        titles=missing
        GC.gc()
    end

    if !isfile("GNN_Julia/doi_dic")
        println("Preparing doi_dic")
        doi_dic = [:doi in collect(keys(dic)) ? dic[:doi] : "no doi" for dic in articles_list_dict]
        Arrow.write("GNN_Julia/doi_dic", Tables.table(doi_dic, header=["doi"]))

        #doi_dic=missing
        #GC.gc()
    end

    if !isfile("GNN_Julia/refs_soft")
        println("Preparing refs_soft")
        refs_soft = [dic[:ref_ids] for dic in articles_list_dict]
        Arrow.write("GNN_Julia/refs_soft", Tables.table(refs_soft, header=["references"]))
        refs_soft=missing
        GC.gc()
    end

    node_features = DataFrame(Arrow.Table("GNN_Julia/node_features"))
    art_soft = DataFrame(Arrow.Table("GNN_Julia/art_soft"))
    paper_id_soft = DataFrame(Arrow.Table("GNN_Julia/paper_id_soft"))
    titles = DataFrame(Arrow.Table("GNN_Julia/titles"))
    doi_dic = DataFrame(Arrow.Table("GNN_Julia/doi_dic"))
    refs_soft = DataFrame(Arrow.Table("GNN_Julia/refs_soft"))


    df = hcat(node_features,art_soft,paper_id_soft,titles,doi_dic,refs_soft)
    Arrow.write("GNN_Julia/df_arrow", df)
    articles_list_dict = missing
end

df= DataFrame(Arrow.Table("GNN_Julia/df_arrow"))
#####################
########## 

if !isfile("GNN_Julia/titles2")

    if length(readdir("crossref_titles/"))==0
        # ========== CONFIGURATION ==========
        const WORKER_COUNT = 4              # Adjust based on your CPU cores
        const BATCH_SIZE = 30              # Memory-safe batch size
        const REQUEST_DELAY = 0.08          # 0.08s delay = ~12.5 req/s per worker
        const USER_AGENT = "YourApp/1.0 (mailto:your@email.com)"  # REPLACE WITH YOUR INFO
        const OUTPUT_DIR = "crossref_titles"  # Folder for saving titles

        # ========== WORKER SETUP ==========
        addprocs(WORKER_COUNT)

        @everywhere begin
            using HTTP, JSON3, URIs, FilePathsBase  # Ensure URIs is included here
            const CROSSREF_URL = "https://api.crossref.org/works"
            const HEADERS = ["User-Agent" => $USER_AGENT]
            const REQ_DELAY = $REQUEST_DELAY
            const OUTPUT_DIR = $OUTPUT_DIR

            # Ensure output directory exists on all workers
            if !isdir(OUTPUT_DIR)
                mkdir(OUTPUT_DIR)
                @info "Created directory: $OUTPUT_DIR on worker $(myid())"
            end

            function fetch_title(doi)
                try
                    # Rate limit first to prevent flooding
                    sleep(REQ_DELAY)
                    
                    encoded_doi = URIs.URI(doi)  # Ensure URIs.URI is used here
                    response = HTTP.get(
                        "$CROSSREF_URL/$encoded_doi", 
                        HEADERS; 
                        readtimeout=30,
                        retry_non_idempotent=true
                    )
                    
                    if response.status == 200
                        data = JSON3.read(response.body)
                        return get(data["message"]["title"], 1, missing)
                    else
                        @warn "Failed to fetch title for DOI: $doi, status: $(response.status)"
                    end
                catch e
                    @warn "Error fetching title for DOI: $doi, Error: $e"
                end
                return missing
            end
        end

        # ========== MAIN PROCESS ==========
        function process_all_dois(doi_list; resume=false)
            # Ensure output directory exists on the main process
            if !isdir(OUTPUT_DIR)
                mkdir(OUTPUT_DIR)
                @info "Created directory: $OUTPUT_DIR on main process"
            end

            # Initialize or resume progress
            processed_dois = Set{String}()
            if resume && isdir(OUTPUT_DIR)
                existing_files = readdir(OUTPUT_DIR)
                processed_dois = Set(basename.(existing_files))  # File names are DOIs
            end
            
            # Configure logging
            logger = SimpleLogger(open("doi_errors.log", "a+"))
            
            # Process in memory-safe batches
            with_logger(logger) do
                total = length(doi_list)
                @info "Starting to process $total DOIs"
                for i in 1:BATCH_SIZE:total
                    batch = doi_list[i:min(i+BATCH_SIZE-1, total)]
                    
                    # Skip already processed DOIs (files)
                    batch = setdiff(batch, processed_dois)
                    isempty(batch) && continue
                    
                    # Parallel fetch
                    @distributed for doi in batch
                        # Check if the file already exists
                        filename = joinpath(OUTPUT_DIR, replace("$doi.txt", "/" => "_"))
                        if isfile(filename)
                            @info "File for DOI: $doi already exists. Skipping API request."
                        else
                            title = fetch_title(doi)
                            if title !== missing
                                @info "Writing title for DOI: $doi to file $filename on worker $(myid())"
                                open(filename, "w") do file
                                    write(file, title)
                                end
                            else
                                @warn "No title found for DOI: $doi"
                            end
                        end
                    end
                    
                    # Progress tracking
                    GC.gc()  # Manual memory cleanup
                    progress = min(i+BATCH_SIZE-1, total)
                    @info "Progress: $(round(100*progress/total, digits=1))% ($progress/$total)"
                end
            end
        end

        # ========== EXECUTION ==========
        # Convert your DOI dictionary to list (replace with actual data)
        doi_urls = filter(row -> row.title == "Not available"  && !isnothing(row.doi), df).doi
        dois = replace.(doi_urls, "https://doi.org/" => "")

        # Start processing (set resume=true to continue from partial results)
        process_all_dois(dois; resume=true)
    end




    println("Preparing titles enriched with crossref")
    # Read the list of files in the output directory
    l_files = readdir("crossref_titles/")
    new_titles = String[]  # Correct initialization for the empty vector
    #df1= filter(row -> row.title == "Not available", df)
    # Loop through the DataFrame
    for i in 1:size(df, 1)  # `range` is unnecessary in this case
        if df.title[i] == "Not available"  && !isnothing(df.doi[i])
            str_file = replace(replace(df.doi[i], "https://doi.org/" => ""), "/" => "_")
            # Check if a file with the DOI name exists in the directory
            if string(str_file, ".txt") in l_files
                # Read the title from the file and push it to `new_titles`
                vs = readlines(open(string("crossref_titles/", str_file, ".txt"), "r"))
                if length(vs)==1
                    push!(new_titles, readlines(open(string("crossref_titles/", str_file, ".txt"), "r"))[1])
                else
                    push!(new_titles, df.title[i])
                end
            else
                push!(new_titles, df.title[i])
            end
        else
            # If title is available, push the title from the dataframe
            push!(new_titles, df.title[i])
        end
    end


    Arrow.write("GNN_Julia/titles2", Tables.table(new_titles, header=["title"]))
end

select!(df, Not(:title))
df.title = DataFrame(Arrow.Table("GNN_Julia/titles2")).title



df.paper_id = df.paper_id_soft
select!(df, Not(:paper_id_soft))
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

df = filter(x->x.title !="Not available", df)
####### BEGIN SECTION NOT USEFUL ANYMORE #######
df.msc_codes_2 = map(y-> unique(map(x -> SubString(x,1:2),y)), df.node_features)
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
    grouped_data_by_paper_id = combine(groupby(df, :paper_id), :software => x -> collect(x) => :software_array, :msc_codes_2 => x -> collect(x) => :msc_array, :title)


    merged_msc_column = [reduce(vcat, vec(pair.first)) for pair in grouped_data_by_paper_id.msc_codes_2_function]
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
unique_paper_ids = Set(unique(df.paper_id))
selected_paper_id = Set(unique(selected_columns.paper_id))

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




CSV.write("software_graph2.edgelist", Tables.table(software_edges); header = false)
