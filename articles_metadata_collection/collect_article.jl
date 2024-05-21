using JSON3
using DataFrames
using JSONTables
using CSV
using HTTP


df = CSV.read("data/full_df.csv", DataFrame)

#println(df[!,"id"])

r = HTTP.request("GET", "https://api.zbmath.org/v1/document/_structured_search?page=0&results_per_page=100&software%20id=11353")
println(r.status)
println(String(r.body))