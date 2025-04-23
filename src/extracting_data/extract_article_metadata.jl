module DataReaderszbMATH
using JSON3
using DataFrames
#using JSONTables
using Arrow
#using StructTypes

function extract_articles_metadata()
    #=
    This function gets .json files from the directory
    return list of dictionaries with the keys :author_name, :database, :datestamp, :document_type,
    :doi, :id, :identifier, :keywords, :language, :msc, :ref_ids, :reviewer_name, :subtitle, :text,
    :title, :year, :zbmath_url
    =#
    files = readdir("./articles_metadata")
    df_list = []
    for file in files
        if endswith(file,".json")
            file_name = string("./data/articles_metadata/", file)
            println(file)
            json_string =read(file_name)
            #println(json_string)
            json_file = JSON3.read(json_string)


            result = json_file.result
            selected_vars = [:contributors, :database, :datestamp, :document_type, :editorial_contributions,
            :id, :identifier, :keywords, :language, :links, :msc, :references,  :title,
            :year, :zbmath_url]
            
            for item in result    
                df_dict = Dict()
                df_dict[:software] = strip(file, ['.','j','s','o','n'])
                for key in item
                    # Create unnested lists with necessary information from the nested json
                    # if value not available from API, save as "Not available"
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
                        df_dict[:language] = lang
                    
                    elseif key.first == :links
                        if key.second != []
                            ident = key.second[1].identifier
                            if ident === nothing
                                doi = nothing
                            else
                                doi = "https://doi.org/" * ident
                            end
                            df_dict[:doi] = doi
                        else
                            df_dict[:doi] = nothing
                        end

                    elseif key.first == :msc
                        df_dict[:msc] = [dic.code for dic in key.second]
                    elseif key.first == :references
                        if key.second != []
                            ref_ids = []
                            for ref in key.second
                                ref_id = ref.zbmath.document_id
                                if ref_id === nothing
                                    continue                    
                                
                                else
                                    ref = ref_id
                                end
                                push!(ref_ids, ref)
                            end
                        else
                            ref_ids = nothing
                        end
                        df_dict[:ref_ids] = ref_ids
                    
                    
                    elseif key.first == :title
                        title = key.second.title
                        if title !== nothing && startswith(title, "zbMATH Open Web Interface contents unavailable")       
                            title = "Not available"
                        end
                        df_dict[:title] = title
                        subtitle = key.second.subtitle
                        if subtitle !== nothing && startswith(subtitle, "zbMATH Open Web Interface contents unavailable")
                            subtitle = "Not available"
                        end
                        df_dict[:subtitle] = subtitle
                    
                    # Create dictionary items for all keys that are not nested
                    elseif in(key.first, selected_vars)
                        df_dict[key.first] = key.second 
                    end
                    
                end
                push!(df_list, df_dict)
            end
        end
    end
    return df_list 
end


function dict_list_to_df(dict_list)
    #=
    Gets dict_list from function before
    creates DataFrame from it
    saves as .csv file
    =#
    df = DataFrame()
    for d in dict_list
        help_dict = Dict()
        for key in d
            first = string(key.first)
            second = key.second
            if second isa Array || second isa JSON3.Array
                help_dict[first] = join(second, "; ")
            elseif second === nothing
                help_dict[first] = string()
            else
                help_dict[first] = string(second)
            end    
        end
        row = DataFrame(help_dict)
        if names(row) != ["author_name", "database", "datestamp", "document_type", "doi", "id", "identifier", "keywords", "language", "msc", "ref_ids", "reviewer_name", "subtitle", "text", "title", "year", "zbmath_url"]
            continue
        end
        append!(df, row)        
    end
    println(names(df))
    println(size(df))
    println(describe(df))
    #CSV.write("./articles_metadata_collection/full_df.csv",df)
    Arrow.write("./data/articles_metadata_collection/full_df_arrow",df)
end

#dict_list = extract_articles_metadata()
#df = dict_list_to_df(dict_list)

end
