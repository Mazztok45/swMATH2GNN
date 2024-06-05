using JSON3
using DataFrames
using JSONTables
using CSV
using HTTP


df = CSV.read("data/full_df.csv", DataFrame)
df = subset(df, :articles_count => a -> a.>0)

for id in df[!,"id"]
    if isfile("articles_metadata_collection/"*string(id)*".json")==false || filesize("articles_metadata_collection/"*string(id)*".json")==0
        
        println(id)
        try
            r = HTTP.request("GET", "https://api.zbmath.org/v1/document/_structured_search?page=0&results_per_page=100&software%20id="*string(id))
            println(r.status)
            #println(String(r.body))
            write("articles_metadata_collection/"*string(id)*".json",String(r.body)) 
        catch y
            println("Exception: "*string(id)) 
        end
    else
        "articles_metadata_collection/"*string(id)*".json already exists"
    end
end

