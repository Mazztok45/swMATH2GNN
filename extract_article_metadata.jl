using JSON3
using DataFrames
using JSONTables

function extract_articles_metadata()
    #files = json_files = readdir("articles_data")
    #for file in files
    json_file = JSON3.read("articles_data/articles_metadata_collection/2.json")
    result = json_file.result

    selected_vars = [:contributors, :database, :datestamp, :document_type, :editorial_contributions,
    :id, :identifier, :keywords, :language, :links, :msc, :references,  :title,
    :year, :zbmath_url]
    df_list = []
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
            
            
            # to be implemted, very brute force but didn't find a better solution
            # because we don't need all the information in the nested dicts.

            elseif key.first == :editorial_contributions
                text = key.second[1].text
                df_dict[:text] = text
                reviewer_name = key.second[1].reviewer.name
                df_dict[:reviewer_name] = reviewer_name

            
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
                df_dict[:msc] = nothing
            
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
        #println(df_dict)
        push!(df_list, df_dict)
    end
    #df = DataFrame(df_list)
    println(df_list)
    return df_list
end


function dict_list_to_df(dict_list)
    df = DataFrame()
    return df
end

        
    #     for k in keys(json_file)
    #         temp_dict = copy(json_file[k])
    #         for var in selected_vars
    #             if temp_dict[var] == nothing
    #                 temp_dict[var] = string()
    #             end
    #             if temp_dict[var] isa Array
    #                 temp_dict[var] = join(temp_dict[var], ";")
    #             end
    #         end
    #         result = Dict{Symbol, Any}()
    
    #         for (k,v) in temp_dict
    #             if k in selected_vars
    #                 push!(result, k=>v)
    #             end
    #         end
    #         temp_df = result |> DataFrame;
    #         append!(df,temp_df)
    # end
    #return result



extract_articles_metadata()