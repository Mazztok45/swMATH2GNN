using TextAnalysis
using MultivariateStats
using SparseArrays
using DataFrames
using LightGraphs
using GraphPlot
module HeteroData

export preprocess_heterodata, keywords_vocabulary, software_to_software_edges, software_to_articles_edges, article_to_article_edges, article_to_article_edges,ids_mapping,create_edge_index
# Sample DataFrames (replace with actual data loading)
#articles_dict = Dict("keywords" => ["Python", "orms"], "id" => Dict(1 => 1001, 2 => 1002),"ref_ids" => Dict(1 => "2", 2 => "1"))
#software_dict = Dict("keywords" => ["Python", "orms"],"id" => Dict(1 => 2001, 2 => 2002),"related_software" => Dict(1 => "2", 2 => "1"),"standard_articles" => Dict(1 => "1", 2 => "2"))

function preprocess_heterodata(articles_dict, software_dict)
    data = Dict()

    # Create nodes for all articles and software with features from the called function
    articles_features, software_features = keywords_vocabulary(articles_dict, software_dict)
    data["article"] = articles_features
    data["software"] = software_features

    # Create edges between software
    software_edges = software_to_software_edges(software_dict)
    data["software_edges"] = software_edges

    # Create edges between software and articles
    sof_art_edges = software_to_articles_edges(software_dict, articles_dict)
    data["sof_art_edges"] = sof_art_edges

    # Create edges between articles
    article_edges = article_to_article_edges(articles_dict)
    data["article_edges"] = article_edges

    println("Built HeteroData Dataset")
    return data
end

function keywords_vocabulary(articles_dict, software_dict)
    software_keywords = String[]
    articles_keywords = String[]
    for value in values(articles_dict["keywords"])
        if isa(value, String)
            value = replace(value, ';' => ' ')
            push!(articles_keywords, value)
        end
    end

    for value in values(software_dict["keywords"])
        if isa(value, String)
            value = replace(value, ';' => ' ')
            push!(software_keywords, value)
        else
            push!(software_keywords, "nan")
        end
    end

    all_keywords = vcat(software_keywords, articles_keywords)

    # Debug: Check all_keywords
    #println("all_keywords: ", all_keywords)

    # Vectorize the keywords using TfIdf
    corpus = Corpus([StringDocument(doc) for doc in all_keywords])
    #println("Corpus created: ", corpus)

    # Create DocumentTermMatrix
    dtm = DocumentTermMatrix(corpus)
    #println("DTM created: ", dtm)

    tfidf_matrix = tf_idf(dtm)
    
    # Debug: Check tfidf_matrix
    #println("tfidf_matrix size: ", size(tfidf_matrix))
    #println("tfidf_matrix type: ", typeof(tfidf_matrix))
    #println("tfidf_matrix: ", tfidf_matrix)

    # Split the tfidf_matrix
    num_software = length(software_keywords)
    num_articles = length(articles_keywords)

    software_features = tfidf_matrix[1:num_software, :]
    articles_features = tfidf_matrix[num_software+1:end, :]

    # Debug: Check split matrices
    #println("software_features size: ", size(software_features))
    #println("articles_features size: ", size(articles_features))

    # Convert sparse matrices to dense matrices for KernelPCA
    software_features_dense = Array(software_features)
    articles_features_dense = Array(articles_features)

    # Check types and shapes
    #println("software_features_dense type: ", typeof(software_features_dense))
    #println("articles_features_dense type: ", typeof(articles_features_dense))
    #println("software_features_dense size: ", size(software_features_dense))
    #println("articles_features_dense size: ", size(articles_features_dense))
    create_edge_index
    # Concatenate the dense matrices vertically
    all_features_dense = vcat(software_features_dense, articles_features_dense)

    # Ensure all_features_dense is a simple 2D array
    #println("all_features_dense type: ", typeof(all_features_dense))
    #println("all_features_dense size: ", size(all_features_dense))

    # Apply KernelPCA for dimensionality reduction
    kpca = fit(KernelPCA, all_features_dense, maxoutdim=5000, kernel=(X, Y) -> X * Y')

    software_features_reduced = MultivariateStats.transform(kpca, software_features_dense)
    articles_features_reduced = MultivariateStats.transform(kpca, articles_features_dense)

    return sparse(software_features_reduced), sparse(articles_features_reduced)
end

function software_to_software_edges(software_dict)
    related_software = software_dict["related_software"]

    for (key, relation) in related_software
        if isa(relation, String)
            ids = parse.(Int, split(relation, r"\s+"))
            related_software[key] = ids
        else
            related_software[key] = Int[]
        end
    end

    mapped_dict = ids_mapping(software_dict)
    edge_index = create_edge_index(related_software, mapped_dict)

    return edge_index
end

function software_to_articles_edges(software_dict, articles_dict)
    standard_articles = software_dict["standard_articles"]

    for (key, relation) in standard_articles
        if isa(relation, String)
            ids = parse.(Int, split(relation, r"\s+"))
            standard_articles[key] = ids
        else
            standard_articles[key] = Int[]
        end
    end

    mapped_dict = ids_mapping(articles_dict)
    edge_index = create_edge_index(standard_articles, mapped_dict)

    return edge_index
end

function article_to_article_edges(articles_dict)
    references = articles_dict["ref_ids"]

    for (key, value) in references
        if isa(value, String)
            ref_ids = parse.(Int, split(value, "; "))
            references[key] = ref_ids
        else
            references[key] = []
        end
    end

    mapped_dict = ids_mapping(articles_dict)
    edge_index = create_edge_index(references, mapped_dict)
    return edge_index
end

function ids_mapping(mapping_dict)
    ids = mapping_dict["id"]
    inverted_dict = Dict(value => key for (key, value) in ids)
    return inverted_dict
end

function create_edge_index(edge_dict, mapped_dict)
    source_indices = Int[]
    target_indices = Int[]

    for (source, targets) in edge_dict
        for target in targets
            if haskey(mapped_dict, target)
                push!(source_indices, source)
                push!(target_indices, mapped_dict[target])
            end
        end
    end

    edge_index = hcat(source_indices, target_indices)'
    return edge_index
end
end
# Example usage
#data = preprocess_heterodata(articles_dict, software_dict)
