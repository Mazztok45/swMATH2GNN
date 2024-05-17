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

json_string = read("data/full_data.json", String)
json_data = Flatten.flatten(json_string)
data = jsontable(json_data)
df = DataFrame(data)


println(size(df))

