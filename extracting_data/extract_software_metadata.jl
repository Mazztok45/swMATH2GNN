using JSON3
using DataFrames
using JSONTables
using CSV


#=
This script reads the .json files from the API, creates a df from it, saves it as CSV
named full_df.csv
=#
df = DataFrame()
json_files = readdir("../data")
selected_vars = [:classification, :description, :articles_count, :authors, :dependencies, :homepage,
:id, :keywords, :license_terms, :name, :operating_systems, :orms_id,:programming_languages,
:related_software, :source_code, :standard_articles, :zbmath_url]


for file in json_files
    if file == "full_df.csv"
        continue
    end

    println(file)
    json_file = JSON3.read(string("../data/", file))
    for k in keys(json_file)
        temp_dict = copy(json_file[k])
        for var in selected_vars
            # convert nothing (nan) values to strings
            if temp_dict[var] == nothing
                temp_dict[var] = string()
            end
            # convert lists to strings with ; in between values
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

CSV.write("../data/full_df.csv", df)