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
#full_df = vcat(df_list)

json_file = JSON3.read("data/0.json")["2"]
json_file = copy(json_file)
json_file[:classification] = join(json_file[:classification],";")


bar = [:classification, :description]
result = Dict{Symbol, Any}()
for (k,v) in json_file
    if k in bar
        push!(result, k=>v)
    end
end
println(result)

df = result |> DataFrame;


####


println(size(df))

