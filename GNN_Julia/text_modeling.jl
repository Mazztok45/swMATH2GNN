using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using DataFrames
include("../extracting_data/extract_software_metadata.jl")
include("../extracting_data/extract_article_metadata.jl")
#using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe
import .DataReaderszbMATH: extract_articles_metadata
using BSON
using GraphNeuralNetworks
using SparseArrays
using Distances
using Base.Threads
filtered_msc = deserialize("filtered_msc.jls")

articles_list_dict = extract_articles_metadata()
software_df  = generate_software_dataframe()


t=DataFrame(permutedims(Matrix(g.target[:,g.ndata.eval_mask])),:auto)

f=DataFrame(permutedims(Matrix(g.features[:,g.ndata.eval_mask])),:auto)

df_keywords = unique(software_df[:,[:id,:keywords]])

BSON.@load "model.bson" model opt


pred=model(g,g.features)

pred = pred .> 0.5
pred=pred[:,g.ndata.eval_mask]

pred=permutedims(pred)

precision = sum((pred .& BitMatrix(Matrix(t)))) / (sum(pred) + 1e-6)

# Recall: count true positives over actual positives
recall = sum((pred .& BitMatrix(Matrix(t))) / (sum(BitMatrix(Matrix(t))) + 1e-6))

# F1-score: harmonic mean of precision and recall
f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)


## 146346 lines
input2pap= hcat(permutedims(filtered_msc), grouped_by_paper_id.paper_id, software_arrays)
## f has has 66777 lines, we need to find the oriig
filtered_msc=nothing
g=nothing

GC.gc()
# Initialize vectors to store the results
vi = Vector{typeof(input2pap[1, 64])}()
vs = Vector{typeof(input2pap[1, 65])}()

# Create dictionaries for faster lookups
input2_dict_64 = Dict((input2pap[line, 1:63]...,) => input2pap[line, 64] for line in 1:size(input2pap, 1))
input2_dict_65 = Dict((input2pap[line, 1:63]...,) => input2pap[line, 65] for line in 1:size(input2pap, 1))

# Iterate through `f` and use the dictionaries to find matches
for line in 1:size(f, 1)
    key = (f[line, 1:63]...,)  # Create a tuple key for the dictionary
    if haskey(input2_dict_64, key)
        push!(vi, input2_dict_64[key])
        push!(vs, input2_dict_65[key])
    end
end

# 66777 lines

#true_input_target=hcat(Matrix(f),vi,vs)




true_output_target=hcat(Matrix(t),vi,vs)
pred_target=hcat(collect(1:size(pred)[1]),pred,vi,vs)






# Example matrices `pred` and `t`
pred = BitMatrix(pred_target[:, 2:69])  # 100 rows, 68 columns
t = BitMatrix(true_output_target[:, 1:68])  # 200 rows, 68 columns

# Function to find the top 10 closest rows in `t` to each row in `pred`, in parallel
function find_closest_rows_parallel(pred, t)
    closest_indices = Vector{Vector{Int}}(undef, size(pred, 1))

    # Use @threads to parallelize the loop
    @threads for i in 1:size(pred, 1)
        current_row = view(pred, i, :)
        # Compute Hamming distances to each row of `t`
        distances = map(j -> hamming(current_row, view(t, j, :)), 1:size(t, 1))
        
        # Get the indices of the 10 smallest distances
        sorted_indices = partialsortperm(distances, 1:100)

        # Store the result
        closest_indices[i] = sorted_indices
    end

    return closest_indices
end

# Find closest rows using the parallel function
closest_rows = find_closest_rows_parallel(pred, t)


#### this where I am 



# Print results for the first row of `pred`
println("Closest 10 rows in `t` for the first row of `pred`: ", closest_rows[1])

soft2keywords = unique(software_df[:,[:keywords,:id]])

# Create a dictionary for fast lookups based on `id` values
id_to_keywords = Dict(row.id => row.keywords for row in eachrow(soft2keywords))

# Initialize result lists
all_list = Vector{Vector{typeof(soft2keywords[1, :keywords])}}()  # For keywords
all_list_s = Vector{Vector{Int64}}()  # For parsed_ids

# Iterate over `closest_rows` and build the unique lists
for l in closest_rows
    lk = Set{typeof(soft2keywords[1, :keywords])}()  # Set to collect unique keywords
    ls = Set{Int64}()  # Set to collect unique parsed_ids

    for ind in l
        val = true_output_target[ind, 70]

        # `val` seems to be iterable, iterate over values and use the dictionary for lookup
        for s in val
            parsed_id = parse(Int64, s)

            # Lookup keywords using the dictionary
            if haskey(id_to_keywords, parsed_id)
                push!(lk, id_to_keywords[parsed_id])
                push!(ls, parsed_id)
            end
        end
    end

    # Convert sets to vectors and push to the result lists
    push!(all_list, collect(lk))  # Collect unique keywords
    push!(all_list_s, collect(ls))  # Collect unique parsed_ids
end

# Assuming all_list_s and true_output_target are defined as in your example

# Convert `true_output_target[:, 70]` to a Vector of Vectors of Int64
true_output_target_ids = [parse.(Int64, x) for x in true_output_target[:, 70]]

# Function to check if two lists of vectors share elements row by row
function check_shared_elements(list1::Vector{Vector{Int64}}, list2::Vector{Vector{Int64}})
    shared_elements = Vector{Bool}(undef, length(list1))

    for i in 1:length(list1)
        shared_elements[i] = !isempty(intersect(list1[i], list2[i]))
    end

    return shared_elements
end

# Compare all_list_s and true_output_target_ids
shared_results = check_shared_elements(all_list_s, true_output_target_ids)



println(sum(shared_results))




paper_title= DataFrame(
    col1=[dic[:id] for dic in articles_list_dict],
    col3=[dic[:keywords] for dic in articles_list_dict],
    col4=[dic[:title] for dic in articles_list_dict]
)

# Create a dictionary for fast lookups based on `id` values
id_to_title = Dict(paper_title.col1[i] => paper_title.col4[i] for i in 1:nrow(paper_title))

# Get titles for each value in `true_output_target[:, 69]`
titles = [id_to_title[val] for val in true_output_target[:, 69] if haskey(id_to_title, val)]

# Print the resulting titles
println(length(titles))

########## cleaning memory 

# Clean variables to free memory
pi_int = nothing
unique_paper_ids = nothing
filtered_data = nothing
grouped_by_paper_id = nothing
software_arrays = nothing
label_to_index = nothing
multi_hot_matrix = nothing
u_soft = nothing
existing_ids = nothing
new_entries = nothing
vec_msc_soft = nothing
label_to_msc = nothing
msc_soft_hot_matrix = nothing
filtered_msc = nothing
software_df = nothing
train_mask = nothing
eval_mask = nothing
test_mask = nothing
sparse_matrix_float = nothing
S = nothing
factorization = nothing
P = nothing
Q = nothing
data = nothing
node_map = nothing
new_x1 = nothing
new_x2 = nothing
reduced_data = nothing
extended_new_x1 = nothing
extended_new_x2 = nothing
artificial_nodes = nothing
X = nothing
y = nothing
X_oversampled = nothing
y_oversampled = nothing
filtered_edges = nothing
input2pap = nothing
input2_dict_64 = nothing
input2_dict_65 = nothing
vi = nothing
vs = nothing
true_output_target = nothing
pred_target = nothing
closest_rows = nothing
soft2keywords = nothing
id_to_keywords = nothing
shared_results = nothing
paper_title = nothing
id_to_title = nothing

# Run garbage collection to free memory
GC.gc()


########## PREDICTION

using TextAnalysis
using Distances
using SparseArrays
using LinearAlgebra
using Base.Threads

using Distances
using SparseArrays
using LinearAlgebra
using Base.Threads

# Simple tokenization using split function for basic word separation
function basic_tokenize(text::String)
    return split(lowercase(text), r"\W+")  # Split by non-word characters and convert to lowercase
end

# Convert titles and keywords list to strings for vectorization
titles_strings = [string(t) for t in titles]
keywords_strings = [join(keywords, ", ") for keywords in all_list]

# Concatenate all text data to fit the TF-IDF model
all_text = vcat(titles_strings, keywords_strings)
# Delete objects that are not needed anymore to free memory
titles = nothing
all_list = nothing
GC.gc()
# Tokenize the text using the simple tokenizer
all_tokens = [basic_tokenize(t) for t in all_text]

# Delete objects that are not needed anymore to free memory
all_text = nothing
GC.gc()

# Create a word-to-index dictionary for all tokens
vocab = unique(vcat(all_tokens...))
vocab_to_index = Dict(word => idx for (idx, word) in enumerate(vocab))

# Delete objects that are not needed anymore to free memory
all_tokens = nothing
GC.gc()

# Create a count matrix manually
count_matrix = spzeros(Float64, length(vocab), length(all_tokens))
for (doc_idx, tokens) in enumerate(all_tokens)
    for token in tokens
        if haskey(vocab_to_index, token)
            word_idx = vocab_to_index[token]
            count_matrix[word_idx, doc_idx] += 1
        end
    end
end

# Delete objects that are no longer needed to free memory
all_tokens = nothing
vocab_to_index = nothing
GC.gc()

# Create a TF-IDF transformer manually
doc_freq = sum(count_matrix .> 0, dims=2)
idf = log.(size(count_matrix, 2) ./ (doc_freq .+ 1))
tfidf_matrix = count_matrix .* idf

# Delete objects that are no longer needed to free memory
count_matrix = nothing
doc_freq = nothing
idf = nothing
GC.gc()

# Convert the TF-IDF matrix to a sparse matrix
tfidf_sparse = sparse(tfidf_matrix)

# Delete the dense TF-IDF matrix to free memory
tfidf_matrix = nothing
GC.gc()

# Split the TF-IDF matrix into title and keyword parts
titles_tfidf = tfidf_sparse[:, 1:length(titles_strings)]
keywords_tfidf = tfidf_sparse[:, length(titles_strings) + 1:end]

# Delete the TF-IDF sparse matrix to free memory
tfidf_sparse = nothing
GC.gc()

# Function to compute cosine similarity between sparse vectors
function cosine_similarity(v1::SparseVector, v2::SparseVector)
    dot_product = dot(v1, v2)
    norm_v1 = norm(v1)
    norm_v2 = norm(v2)
    return dot_product / (norm_v1 * norm_v2 + eps())
end

# Convert title and keyword TF-IDF matrices to SparseVectors for each column
titles_tfidf_vectors = [view(titles_tfidf, :, i) for i in 1:size(titles_tfidf, 2)]
keywords_tfidf_vectors = [view(keywords_tfidf, :, i) for i in 1:size(keywords_tfidf, 2)]

# Delete title and keyword TF-IDF matrices to free memory
titles_tfidf = nothing
keywords_tfidf = nothing
GC.gc()

# Predict the closest software set for each title using cosine similarity (parallelized)
predicted_software_sets = Vector{Any}(undef, length(titles_tfidf_vectors))

@threads for i in 1:length(titles_tfidf_vectors)
    title_vec = sparse(titles_tfidf_vectors[i])  # Convert to SparseVector

    similarities = [cosine_similarity(title_vec, sparse(keywords_tfidf_vectors[j])) for j in 1:length(keywords_tfidf_vectors)]

    # Find the index of the highest similarity
    max_idx = argmax(similarities)
    predicted_software_sets[i] = all_list_s[max_idx]
end

# Delete SparseVectors used for TF-IDF to free memory
titles_tfidf_vectors = nothing
keywords_tfidf_vectors = nothing
GC.gc()

# Calculate F1 Score Manually
# Convert the predicted software sets and true software sets to sets for F1 calculation
predicted_sets = [Set(predicted) for predicted in predicted_software_sets]
true_sets = [Set(true_vals) for true_vals in true_output_target_ids]

# Delete predicted_software_sets to free memory
predicted_software_sets = nothing
GC.gc()

# Compute F1 score for each set and take the average
f1_scores = []

@threads for i in 1:length(predicted_sets)
    true_vals = true_sets[i]
    predicted_vals = predicted_sets[i]
    true_positive = length(intersect(true_vals, predicted_vals))
    false_positive = length(predicted_vals) - true_positive
    false_negative = length(true_vals) - true_positive

    # Calculate precision, recall, and F1 score
    precision = true_positive / (true_positive + false_positive + eps())
    recall = true_positive / (true_positive + false_negative + eps())

    # Avoid division by zero
    f1 = if precision + recall == 0
        0.0
    else
        2 * ((precision * recall) / (precision + recall))
    end

    push!(f1_scores, f1)
end

# Delete sets used for F1 calculation to free memory
predicted_sets = nothing
true_sets = nothing
GC.gc()

# Print the average F1 score
average_f1_score = mean(f1_scores)
println("Average F1 Score: ", average_f1_score)
