using CSV
using DataFrames
using GraphNeuralNetworks
using Flux
using TextAnalysis
using MultivariateStats
using SparseArrays

include("hetero_data.jl")
using .HeteroData

# Ensure the fit function is imported from MultivariateStats
import MultivariateStats: fit, KernelPCA

# Helper functions to handle Missing values and ensure correct types
function convert_to_string(x)
    return !ismissing(x) ? string(x) : ""
end

function convert_to_nothing(x)
    return !ismissing(x) ? x : nothing
end

function convert_to_default_string(x, default)
    return !ismissing(x) ? string(x) : default
end

# Define the Article structure based on inferred schema
struct Article
    author_name::Union{String, Nothing}
    database::String
    datestamp::String
    document_type::String
    doi::Union{String, Nothing}
    id::String
    identifier::String
    keywords::String
    language::String
    msc::String
    ref_ids::Union{String, Nothing}
    reviewer_name::Union{String, Nothing}
    subtitle::Union{String, Nothing}
    text::Union{String, Nothing}
    title::String
    year::Int64
    zbmath_url::String
end

# Define the Software structure based on inferred schema
struct Software
    articles_count::Int64
    authors::String
    classification::Union{String, Nothing}
    dependencies::Union{String, Nothing}
    description::Union{String, Nothing}
    homepage::String
    id::String
    keywords::String
    license_terms::Union{String, Nothing}
    name::String
    operating_systems::Union{String, Nothing}
    orms_id::Union{String, Nothing}
    programming_languages::Union{String, Nothing}
    related_software::String
    source_code::Union{String, Nothing}
    standard_articles::Union{String, Nothing}
    zbmath_url::String
end

# Function to create Article from DataFrame row
function create_article(row)
    return Article(
        convert_to_nothing(row.author_name),
        row.database,
        row.datestamp,
        row.document_type,
        convert_to_nothing(row.doi),
        convert_to_string(row.id),
        convert_to_string(row.identifier),
        convert_to_string(row.keywords),
        row.language,
        row.msc,
        convert_to_nothing(row.ref_ids),
        convert_to_nothing(row.reviewer_name),
        convert_to_nothing(row.subtitle),
        convert_to_nothing(row.text),
        row.title,
        row.year,
        row.zbmath_url
    )
end

# Function to create Software from DataFrame row
function create_software(row)
    return Software(
        row.articles_count,
        convert_to_string(row.authors),
        convert_to_nothing(row.classification),
        convert_to_nothing(row.dependencies),
        convert_to_nothing(row.description),
        row.homepage,
        convert_to_string(row.id),
        convert_to_string(row.keywords),
        convert_to_nothing(row.license_terms),
        row.name,
        convert_to_nothing(row.operating_systems),
        convert_to_string(row.orms_id),
        convert_to_nothing(row.programming_languages),
        convert_to_default_string(row.related_software, "no software"),
        convert_to_nothing(row.source_code),
        convert_to_nothing(row.standard_articles),
        row.zbmath_url
    )
end

# Load full data into DataFrames
articles_df = CSV.read("./articles_metadata_collection/full_df.csv", DataFrame)
software_df = CSV.read("./data/full_df.csv", DataFrame)

# Print the column names of the DataFrames to verify
println("Articles DataFrame columns: ", names(articles_df))
println("Software DataFrame columns: ", names(software_df))

# Convert DataFrames to dictionaries with struct types
articles_dict = Dict(articles_df.id[i] => create_article(articles_df[i, :]) for i in 1:size(articles_df, 1))
software_dict = Dict(software_df.id[i] => create_software(software_df[i, :]) for i in 1:size(software_df, 1))

# Print keys to verify dictionaries
println("Articles Dict Keys: ", keys(articles_dict))
println("Software Dict Keys: ", keys(software_dict))

# Check for the presence of "related_software" key in software_dict
if !haskey(software_dict, "related_software")
    println("Warning: 'related_software' key not found in software_dict.")
end

# Process the data into a HeteroData() object
try
    data = preprocess_heterodata(articles_dict, software_dict)
    println("Data preprocessing successful")
catch e
    println("Error in preprocess_heterodata: ", e)
end

# Print data metadata if available
if isdefined(Main, :data) && data !== nothing
    metadata = data.metadata
    println(metadata)
end

# Define the transformation (Random
