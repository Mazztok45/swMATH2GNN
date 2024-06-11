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
                df_dict["author_name"] = author_names
            
            elseif key.first == :document_type
                doc_type = key.second.description
                df_dict["document_type"] = doc_type
            
            
            # to be implemted, very brute force but didn't find a better solution
            # because we don't need all the information in the nested dicts.

            elseif key.first == :editorial_contributions
            
            elseif key.first == :language
            
            elseif key.first == :links
            
            elseif key.first == :msc
            
            elseif key.first == :references
            
            elseif key.first == :language
            
            elseif key.first == :title
            
            # else, all the keys that are not nested
            elseif in(key.first, selected_vars)
                df_dict[key.first] = key.second 
            end
            
        end
        println(df_dict)
        push!(df_list, df_dict)
    end
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