using JSON3
using DataFrames
using JSONTables
using LinearAlgebra
using Flatten
json_files = readdir("data")

#df_list = []
"""
for file in json_files
    json_string = read(string("data/", file), String)
    data = JSON.parse(json_string)
    df = DataFrame(data)
    t_df = DataFrame(Matrix(df)')
    println(names(df))
    push!(df_list, df)
end
"""



### Variables to further add in selected_vars, step by step, some are complex

#:dependencies
#:homepage
#:idl
#:keywords
#:license_terms
#:name
#:operating_systems
#:orms_id
#:programming_languages
#:related_software
#:source_code
#:standard_articles
#:zbmath_url


df = DataFrame()
selected_vars = [:classification, :description, :articles_count, :authors, :dependencies]

json_file = JSON3.read("data/0.json")
print(json_file)
for k in keys(json_file)
    temp_dict = copy(json_file[k])
    temp_dict[:classification] = join(temp_dict[:classification],";")
    if temp_dict[:dependencies] == nothing
        temp_dict[:dependencies] = string()
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


println(df)