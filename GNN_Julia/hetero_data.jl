module HeteroDataProcessing

using LightGraphs
using DataFrames
using CSV
using TextAnalysis
using MultivariateStats
using SparseArrays

export preprocess_heterodata
import TextAnalysis: tf_idf, DocumentTermMatrix, StringDocument
# Function to preprocess the hetero data
function preprocess_heterodata(articles_dict::Dict, software_dict::Dict)
    data = Dict{Symbol, Any}()

    # Create nodes for all articles and software with the features from the called function
    articles_features, software_features = keywords_vocabulary(articles_dict,software_dict)
    data[:article] = articles_features
    data[:software] = software_features

    # Create edges between the software
    software_edges = software_to_software_edges(software_dict)
    data[:software_related_software] = software_edges

    # Create edges between software and articles
    sof_art_edges = software_to_articles_edges(software_dict, articles_dict)
    data[:software_mentioned_in_article] = sof_art_edges

    # Create edges between article and article
    article_edges = article_to_article_edges(articles_dict)
    data[:article_references_article] = article_edges

    println("Built HeteroData Dataset")
    return data
end

function keywords_vocabulary(articles_dict::Dict, software_dict::Dict)
    software_keywords = []
    articles_keywords = []

    for value in values(articles_dict)
        if typeof(value.keywords) == String
            if startswith(value.keywords, "zbMATH Open Web Interface contents unavailable due to conflicting licenses.")
                push!(articles_keywords, "not_available")
            else
                elem = replace(value.keywords, ';' => ' ')
                push!(articles_keywords, elem)
            end
        end
    end

    for value in values(software_dict)
        if typeof(value.keywords) == String
            if startswith(value.keywords, "zbMATH Open Web Interface contents unavailable due to conflicting licenses.")
                push!(software_keywords, "not_available")
            else
                elem = replace(value.keywords, ';' => ' ')
                push!(software_keywords, elem)
            end
        else
            push!(software_keywords, "nan")
        end
    end

    all_keywords = vcat(software_keywords, articles_keywords)

    
    ### Preparing the corpus
    list_soft_crps=[]
    for text in software_keywords
        println(typeof(text))
        push!(list_soft_crps, StringDocument(text))
    end
    soft_crps = Corpus(list_soft_crps)

    list_art_crps=[]
    for text in articles_keywords
        println(typeof(text))
        push!(list_art_crps, StringDocument(text))
    end
    art_crps = Corpus(list_art_crps)
    #vectorizer = TFIDF()
    #fit!(vectorizer, all_keywords)
    software_features = tf_idf(DocumentTermMatrix(soft_crps))
    articles_features = tf_idf(DocumentTermMatrix(art_crps))


    # Articles features dimensionality reduction
    pca = PCA(5000)
    articles_features_reduced = pca(articles_features)

    # Software features dimensionality reduction
    software_features_reduced = pca(software_features)

    return articles_features_reduced, software_features_reduced
end

function software_to_software_edges(software_dict::Dict)
    related_software = software_dict["related_software"]

    for (key, relation) in related_software
        if typeof(relation) == String
            ids = parse.(Int, matchall(r":id\s*=>\s*\d+", relation))
            related_software[key] = ids
        else
            related_software[key] = []
        end
    end

    mapped_dict = ids_mapping(software_dict)
    edge_index = create_edge_index(related_software, mapped_dict)

    return edge_index
end

function software_to_articles_edges(software_dict::Dict, articles_dict::Dict)
    standard_articles = software_dict["standard_articles"]

    for (key, relation) in standard_articles
        if typeof(relation) == String
            ids = parse.(Int, matchall(r":id\s*=>\s*\d+", relation))
            standard_articles[key] = ids
        else
            standard_articles[key] = []
        end
    end

    mapped_dict = ids_mapping(articles_dict)
    edge_index = create_edge_index(standard_articles, mapped_dict)

    return edge_index
end

function article_to_article_edges(articles_dict::Dict)
    references = articles_dict["ref_ids"]

    for (key, value) in references
        if typeof(value) == String
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

function ids_mapping(mapping_dict::Dict)
    ids = mapping_dict["id"]
    inverted_dict = Dict(value => key for (key, value) in pairs(ids))
    return inverted_dict
end

function create_edge_index(edge_dict::Dict, mapped_dict::Dict)
    source_indices = []
    target_indices = []

    for (source, targets) in edge_dict
        for target in targets
            if haskey(mapped_dict, target)
                push!(source_indices, source)
                target_map = mapped_dict[target]
                push!(target_indices, target_map)
            end
        end
    end

    source_tensor = collect(source_indices)
    target_tensor = collect(target_indices)

    edge_index = [source_tensor target_tensor]'

    return edge_index
end

end