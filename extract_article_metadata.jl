using JSON3
using DataFrames
using JSONTables
using CSV

function extract_articles_metadata()
    files = readdir("articles_data/articles_metadata_collection")
    df_list = []
    for file in files
        file_name = string("articles_data/articles_metadata_collection/", file)
        json_file = JSON3.read(file_name)

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