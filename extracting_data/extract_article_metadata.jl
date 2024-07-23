using JSON3
using DataFrames
using JSONTables
using CSV
using StructTypes


struct Reviewer
    author_code::Union{Nothing, String}
    reviewer_id::Union{Nothing, String}
    name::Union{Nothing, String}
    sign::Union{Nothing, String}
end

struct EditorialContribution
    language::String
    reviewer::Reviewer
    text::String
    contribution_type::String
end

struct DocumentType
    code::String
    description::String
end

struct Author
    aliases::Vector{String}
    checked::String
    codes::Vector{String}
    name::String
end

struct Contributors
    authors::Vector{Author}
    author_references::Vector{Any}
    editors::Vector{Any}
end

struct Language
    languages::Vector{String}
    addition::Vector{Union{Nothing, String}}
end

struct Link
    identifier::String
    type::String
    url::Union{Nothing, String}
end

struct Msc
    code::String
    scheme::String
    text::String
end

struct Issn
    number::String
    type::String
end

struct Series
    acronym::Union{Nothing, String}
    issn::Vector{Issn}
    issue::Union{Nothing, String}
    issue_id::Int
    parallel_title::Union{Nothing, String}
    part::Union{Nothing, String}
    publisher::String
    series_id::Int
    short_title::String
    title::String
    volume::String
    year::String
end

struct Source
    book::Vector{Any}
    pages::String
    series::Vector{Series}
    source::String
end

struct Title
    addition::Union{Nothing, String}
    original::Union{Nothing, String}
    subtitle::Union{Nothing, String}
    title::String
end

struct Result
    biographic_references::Vector{Any}
    contributors::Contributors
    database::String
    datestamp::String
    document_type::DocumentType
    editorial_contributions::Vector{EditorialContribution}
    id::Int
    identifier::String
    keywords::Vector{Union{Nothing, String}}
    language::Language
    license::Vector{Any}
    links::Vector{Link}
    msc::Vector{Msc}
    references::Vector{Any}
    source::Source
    states::Vector{Vector{String}}
    title::Title
    year::String
    zbmath_url::String
end

struct Status
    execution::String
    execution_bool::Bool
    internal_code::String
    last_id::Union{Nothing, String}
    nr_total_results::Int
    nr_request_results::Int
    query_execution_time_in_seconds::Float64
    status_code::Int
    time_stamp::String
end

struct JSONStructure
    result::Vector{Result}
    status::Status
end

StructTypes.StructType(::Type{JSONStructure}) = StructTypes.Struct()

##
function extract_articles_metadata()
    files = readdir("articles_data/articles_metadata_collection")
    df_list = []
    for file in files
        file_name = string("articles_data/articles_metadata_collection/", file)
        #println(file_name)
        json_string =read(file_name)
        #println(json_string)
        json_file = JSON3.read(json_string, JSONStructure)


        result = json_file.result
        selected_vars = [:contributors, :database, :datestamp, :document_type, :editorial_contributions,
        :id, :identifier, :keywords, :language, :links, :msc, :references,  :title,
        :year, :zbmath_url]
        for item in result    
            df_dict = Dict()
            for key in item
                if key.first == :contributors
                    contrib = key.second
                    authors = contrib.authors
                    author_names = []
                    for author in authors
                        name = author.name
                        if startswith(name, "zbMATH Open Web Interface contents unavailable")
                            name = "Not available"
                        end
                        push!(author_names, name)
                    end
                    df_dict[:author_name] = author_names
                
                elseif key.first == :document_type
                    doc_type = key.second.description
                    df_dict[:document_type] = doc_type

                elseif key.first == :editorial_contributions
                    if key.second != []
                        text = key.second[1].text
                        df_dict[:text] = text
                        reviewer_name = key.second[1].reviewer.name
                        df_dict[:reviewer_name] = reviewer_name
                    else
                        df_dict[:reviewer_name] = nothing
                    end

                elseif key.first == :language
                    lang =  key.second.languages
                    df_dict["language"] = lang
                
                elseif key.first == :links
                    if key.second != []
                        ident = key.second[1].identifier
                        if ident == nothing
                            doi = nothing
                        else
                            doi = "https://doi.org/" * ident
                        end
                        df_dict[:doi] = doi
                    else
                        df_dict[:doi] = nothing
                    end

                elseif key.first == :msc
                    df_dict[:msc] = string(key.second)
                
                elseif key.first == :references
                    if key.second != []
                        ref_dois = []
                        for ref in key.second
                            ref_doi = ref.doi
                            if ref_doi == nothing
                                continue                        
                            elseif startswith(ref_doi, "zbMATH Open Web Interface contents unavailable")
                                continue
                            else
                                ref = ref_doi
                            end
                            push!(ref_dois, ref)
                        end
                    else
                        ref_dois = nothing
                    end
                    df_dict[:ref_dois] = ref_dois
                
                
                elseif key.first == :title
                    title = key.second.title
                    if title != nothing && startswith(title, "zbMATH Open Web Interface contents unavailable")       
                        title = "Not available"
                    end
                    df_dict[:title] = title
                    subtitle = key.second.subtitle
                    if subtitle != nothing && startswith(subtitle, "zbMATH Open Web Interface contents unavailable")
                        subtitle = "Not available"
                    end
                    df_dict[:subtitle] = subtitle
                
                # else, all the keys that are not nested
                elseif in(key.first, selected_vars)
                    df_dict[key.first] = key.second 
                end
                
            end
            push!(df_list, df_dict)
        end
    end
    return df_list 
end


# TODO: language, keywords "not available" solution

function dict_list_to_df(dict_list)
    df = DataFrame()
    for d in dict_list
        help_dict = Dict()
        for key in d
            first = string(key.first)
            second = key.second
            if second isa Array || second isa JSON3.Array
                help_dict[first] = join(second, "; ")
            elseif second == nothing
                help_dict[first] = string()
            else
                help_dict[first] = string(second)
            end    
        end
        row = DataFrame(help_dict)
        if names(row) != ["author_name", "database", "datestamp", "document_type", "doi", "id", "identifier", "keywords", "language", "msc", "ref_dois", "reviewer_name", "subtitle", "text", "title", "year", "zbmath_url"]
            continue
        end
        append!(df, row)        
    end
    println(names(df))
    println(size(df))
    CSV.write("articles_data/full_df.csv",df)
end

dict_list = extract_articles_metadata()
df = dict_list_to_df(dict_list)