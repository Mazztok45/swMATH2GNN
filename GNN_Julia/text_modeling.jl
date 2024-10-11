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
using Serialization
filtered_msc = deserialize("filtered_msc.jls")

articles_list_dict = extract_articles_metadata()
software_df  = generate_software_dataframe()
## 146346 lines
input2pap= hcat(permutedims(filtered_msc), grouped_by_paper_id.paper_id, software_arrays)
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
# Track unique identifiers using a built-in set
unique_identifiers = Set{Int}()
for i in 1:length(eval_mask)
    if eval_mask[i] == 0
        new_values[i] = nothing
    elseif eval_mask[i] == 1
        identifier = vi[vi_index]
        vi_index += 1
        if !in(identifier, unique_identifiers)
            new_values[i] = identifier
            new_eval_mask[i] = 1
            push!(unique_identifiers, identifier)
        else
            new_values[i] = identifier
            new_eval_mask[i] = 0
        end
    end
end

# Step 2: Verify the sum of new_eval_mask is equal to the number of unique values in vi
@assert sum(new_eval_mask) == length(unique(vi)) == 3938
#true_input_target=hcat(Matrix(f),vi,vs)




t=DataFrame(permutedims(Matrix(g.target[:,BitVector(new_eval_mask)])),:auto)

f=DataFrame(permutedims(Matrix(g.features[:,BitVector(new_eval_mask)])),:auto)

df_keywords = unique(software_df[:,[:id,:keywords]])

BSON.@load "model.bson" model opt


pred=model(g,g.features)

pred = pred .> 0.1
pred=pred[:,BitVector(new_eval_mask)]

pred=permutedims(pred)

precision = sum((pred .& BitMatrix(Matrix(t)))) / (sum(pred) + 1e-6)

# Recall: count true positives over actual positives
recall = sum((pred .& BitMatrix(Matrix(t))) / (sum(BitMatrix(Matrix(t))) + 1e-6))

# F1-score: harmonic mean of precision and recall
f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
# True positives (TP) and true negatives (TN)
true_positives = sum(pred .& BitMatrix(Matrix(t)))
true_negatives = sum(.~pred .& .~BitMatrix(Matrix(t)))

# Total number of elements
total_elements = length(pred)

# Accuracy: (TP + TN) / total
accuracy = (true_positives + true_negatives) / (total_elements + 1e-6)

println("Metrics:\n - Accuracy: $accuracy\n - Precision: $precision\n - Recall: $recall\n - F1 Score: $f1_score")



#############
# Assuming vi is the array with 66777 identifiers
# and g.ndata.eval_mask is the array with 292692 rows of 0 and 1

# The `new_values` array has 292692 rows, containing `Nothing` for 0 in eval_mask,
# and identifiers for where 1 was in eval_mask, ensuring no duplicate identifiers.
# The `new_eval_mask` ensures only unique identifiers are marked with 1.

#############
#Graph

dfs=unique(software_df[:,[:id,:classification]])
# Create a new column for classifications as an array of strings
dfs.classification_split = [split(x, ";") for x in dfs.classification]
 # Flatten the list of classifications and convert it into a DataFrame
 classification_list = vcat(dfs.classification_split...)
 # Flatten the list of classifications and convert it into a DataFrame
 classification_counts = DataFrame(classification=classification_list)
# Count the occurrences of each classification
classification_summary = combine(groupby(classification_counts, :classification), nrow => :count)


# Sort the DataFrame by classification in ascending order
sorted_classification_summary = sort(classification_summary, :classification)

# Extract data from the sorted DataFrame
classifications = sorted_classification_summary.classification
counts = sorted_classification_summary.count

# Determine appropriate y-tick values
yticks_values = 0:2000:maximum(counts)

# Plot a bar chart of classification counts
bar_plot = bar(classifications, counts, 
    xlabel = "MSC Code", 
    ylabel = "Software count", 
    title = "Distribution of Software Count per MSC Code", 
    legend = false, 
    xtickfont = font(8), 
    rotation = 45,
    yticks = (yticks_values, [string(v) for v in yticks_values]))  # Set y-ticks explicitly with readable labels

# Display the plot

# Save the plot to a file (e.g., PNG)
savefig(bar_plot, "msc_code_distribution.png")

# You can also use other formats like:
# savefig(bar_plot, "msc_code_distribution.pdf")
# savefig(bar_plot, "msc_code_distribution.svg")

#############

## f has has 66777 lines, we need to find the oriig
#filtered_msc=nothing
#g=nothing

GC.gc()
# Initialize vectors to store the results




true_output_target=hcat(Matrix(t),vi,vs)
pred_target=hcat(collect(1:size(pred)[1]),pred,vi,vs)






# Example matrices `pred` and `t`
#pred = BitMatrix(pred_target[:, 2:69])  # 100 rows, 68 columns
#t = BitMatrix(true_output_target[:, 1:68])  # 200 rows, 68 columns

# Function to find the top 10 closest rows in `t` to each row in `pred`, in parallel
function find_closest_rows_parallel(pred, t)
    closest_indices = Vector{Vector{Int}}(undef, size(pred, 1))

    # Use @threads to parallelize the loop
    @threads for i in 1:size(pred, 1)
        current_row = view(pred, i, :)
        # Compute Hamming distances to each row of `t`
        distances = map(j -> hamming(current_row, view(t, j, :)), 1:size(t, 1))
        
        # Get the indices of the 10 smallest distances
        sorted_indices = partialsortperm(distances, 1:500)

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

unique(sort!(DataFrame(hcat(true_output_target[:, 69],true_output_target_ids, all_list_s), :auto)))

df_true=unique(DataFrame(true_output_target[:,69:70],:auto))


shared_results = check_shared_elements(closest_rows, df_true.x2)

hcat_shared_result=hcat(true_output_target[:, 69],shared_results )

df_u_hcat=DataFrame(hcat_shared_result,:auto)
df_u_hcat=unique(df_u_hcat)
println(sum(df_u_hcat.x2)/size(df_u_hcat)[1])





function check_all_elements_included(list1::Vector{Vector{Int64}}, list2::Vector{Vector{Int64}})
    all_included = Vector{Bool}(undef, length(list1))

    for i in 1:length(list1)
        all_included[i] = all(x -> x in list1[i], list2[i])
    end

    return all_included
end

# Compare all_list_s and true_output_target_ids
all_included_results = check_all_elements_included(all_list_s, true_output_target_ids)

println(sum(all_included_results))

hcat_shared_result=hcat(true_output_target[:, 69],all_included_results)

df_u_hcat=DataFrame(hcat_shared_result,:auto)
df_u_hcat=unique(df_u_hcat)
println(sum(df_u_hcat.x2)/size(df_u_hcat)[1])


##### UNTIL HERE 75% of the articles have a software set included in the software set previously identified

###### RFERENCES SECTION


##########
using HTTP, Gumbo, Cascadia

function extract_references(url::String)
    # Get the HTML content of the page
    response = HTTP.get(url; require_ssl_verification=false)
    html_content = String(response.body)

    # Parse the HTML content
    parsed_html = parsehtml(html_content)

    # Select all `<td>` elements with class "space" that contain references
    references_cells = eachmatch(Selector("td.space"), parsed_html.root)
    references = String[]

    # Extract reference text from each selected `<td>` element
    for (index, cell) in enumerate(references_cells)
        # Extracting all text content within the cell's children
        if index > 2
            println(index)
            reference_text = Gumbo.text(cell)

            push!(references, strip(reference_text))
        end
    end

    return references
end

# Example usage
url = "https://zbmath.org/7110436"
references = extract_references(url)

using Base.Threads

function extract_references(url::String)
    try
        # Get the HTML content of the page
        response = HTTP.get(url; require_ssl_verification=false)
        html_content = String(response.body)

        # Parse the HTML content
        parsed_html = parsehtml(html_content)

        # Select all `<td>` elements with class "space" that contain references
        references_cells = eachmatch(Selector("td.space"), parsed_html.root)
        references = String[]

        # Extract reference text from each selected `<td>` element
        for cell in references_cells
            # Extracting all text content within the cell's children
            reference_text = Gumbo.text(cell)
            push!(references, strip(reference_text))
        end

        return references
    catch e
        println("Error accessing URL $url: $e")
        return String[]  # Return an empty array in case of error
    end
end


function extract_references_parallel(doc_ids::Vector{String})
    # Create a container to store references for each document
    pap2ref = Vector{Vector{String}}(undef, length(doc_ids))

    # Perform parallel extraction using `@threads` macro
    @threads for i in 1:length(doc_ids)
        doc_id = doc_ids[i]
        url = "https://zbmath.org/" * doc_id
        println("Thread $(threadid()): Processing document ID: $doc_id ($i of $(length(doc_ids)))")  # Log progress
        references = extract_references(url)
        pap2ref[i] = references
        println("Thread $(threadid()): Finished processing document ID: $doc_id")  # Log completion
    end

    return pap2ref
end

# Example usage

doc_ids = string.(true_output_target[:, 69])  # Convert each element to String
references_data = extract_references_parallel(doc_ids)
println(references_data)

using Arrow, DataFrames

# Assuming references_data and true_output_target have already been extracted
paper_ids = true_output_target[:, 69]

# Create flat lists of paper IDs and corresponding reference titles
paper_ids_flat = []
reference_titles = []

for (i, references) in enumerate(references_data)
    for reference in references
        push!(paper_ids_flat, paper_ids[i])
        push!(reference_titles, reference)
    end
end

# Construct DataFrame
df = DataFrame(paper_id = paper_ids_flat, reference_title = reference_titles)

# Save the DataFrame to an Arrow file
Arrow.write("references_data.arrow", df)

##########
######

paper_title= DataFrame(
    col1=[dic[:id] for dic in articles_list_dict],
    col3=[dic[:keywords] for dic in articles_list_dict],
    col4=[dic[:title] for dic in articles_list_dict]
)


text=[
    (dic[:id], get(dic, :text, get(dic, :keywords, nothing)))
    for dic in articles_list_dict
    if ((:text in keys(dic) && dic[:text] != "zbMATH Open Web Interface contents unavailable due to conflicting licenses.") ||
        (:keywords in keys(dic) && dic[:keywords] != "zbMATH Open Web Interface contents unavailable due to conflicting licenses.")) &&
       get(dic, :text, get(dic, :keywords, nothing)) != "zbMATH Open Web Interface contents unavailable due to conflicting licenses."
]


println(size(text))

common_elements = intersect(true_output_target[:, 69], ids_in_text)






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
