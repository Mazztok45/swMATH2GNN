using JSON3
using DataFrames
using JSONTables

function extract_articles_metadata()
    #files = json_files = readdir("articles_data")
    #for file in files
    json_file = JSON3.read("articles_data/articles_metadata_collection/2.json")
    result = json_file.result

    selected_vars = [:biographic_references, :contributors, :database, :datestamp, :document_type, :editorial_contributions,
    :id, :identifier, :keywords, :language, :license, :links, :msc, :references, :source, :states, :title,
    :year, :zbmath_url]
    for (key, value) in result
        df_dict = Dict()
        #temp_dict = copy(result[k])
        for var in selected_vars
            if var == "contributors"
                print(value)
            end
            #println(var)
            #print(result[k][var])
        end
        #println(df_dict)

        result |> DataFrame;
        #append!(df,temp_df)
        #println(result)
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