using JSON3
using DataFrames
using JSONTables
using CSV
using Graphs
using GraphPlot
using LinearAlgebra
using MetaGraphsNext
using Compose
import Cairo, Fontconfig

function extract_metadata()
    
    df = DataFrame()
    json_file = JSON3.read("data/0.json")
    selected_vars = [:classification, :description, :articles_count, :authors, :dependencies, :homepage,
    :id, :keywords, :license_terms, :name, :operating_systems, :orms_id,:programming_languages,
    :related_software, :source_code, :standard_articles, :zbmath_url]
      
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
    
    return df
end

function create_metagraph()
    software_df = extract_metadata()
    println("extracted metadata!")
    software_rel = MetaGraph(
        Graph();
        label_type = Int, 
        vertex_data_type=String,
        # edge_data_type=Nothing,  
        graph_data="Relations between softwares"
    )
    #creating the nodes
    for row in eachrow(software_df)
        software_rel[row."id"] = row."name"
    end
    #creating the edges
    for row in eachrow(software_df)
        id_row = row."id"
        rel_string = row."related_software"
        name_row = row."name"

        for label in labels(software_rel)
            name = software_rel[label]
            if occursin(name, rel_string)
                edge =add_edge!(software_rel, label, id_row)
                
            end

        end
    end

    draw(PNG("software_rel.png", 16cm, 16cm), gplot(software_rel, layout=spectral_layout))
    return software_rel
end

software_rel = create_metagraph()

# TODO: -try out with big Dataset
#       - find a way to display & save the graph
