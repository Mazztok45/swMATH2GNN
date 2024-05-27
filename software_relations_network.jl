using JSON3
using DataFrames
using JSONTables
using CSV
using Graphs
using GraphPlot
using LinearAlgebra
using MetaGraphsNext

function extract_metadata()
    #TODO: Make a sample Dataset to try out the Graphs
    df = DataFrame()
    json_files = readdir("data")
    selected_vars = [:classification, :description, :articles_count, :authors, :dependencies, :homepage,
    :id, :keywords, :license_terms, :name, :operating_systems, :orms_id,:programming_languages,
    :related_software, :source_code, :standard_articles, :zbmath_url]
    
    
    for file in json_files
        if file == "full_df.csv"
            continue
        end
        println(file)
        json_file = JSON3.read(string("data/", file))
        for k in keys(json_file)
            temp_dict = copy(json_file[k])
            for var in selected_vars
                if temp_dict[var] == nothing
                    temp_dict[var] = string()
                end
                if temp_dict[var] isa Array
                    temp_dict[var] = join(temp_dict[var], ";")
                end
            end
            result = Dict{Symbol, Any}()
    
            for (k,v) in temp_dict
                if k in selected_vars
                    push!(result, k=>v)
                end
            end
            temp_df = result |> DataFrame;
            append!(df,temp_df)
        end
    end
    return df
end

function create_metagraph()
    software_df = extract_metadata()
    println("extracted metadata!")
    software_rel = MetaGraph(
        Graph();
        label_type = Symbol, 
        vertex_data_type=Dict,
        edge_data_type=Symbol,  
        graph_data="Relations between softwares"
    )
    for row in eachrow(software_df)
        software_rel[:row.name] = row.id
    end
    return nv(software_rel)
end

software_rel = create_metagraph()
print(software_rel)