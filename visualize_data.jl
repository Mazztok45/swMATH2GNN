using CSV
using Plots
using DataFrames
using JSON3
using JSONTables

#I'm going to use just the first file to try out some visualisations, for practical reasons.

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



