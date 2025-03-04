using HTTP
using JSON

function fetch_author_data()
    # Define the base URL
    base_url = "https://api.zbmath.org/v1/author/_structured_search"

    # Define the query parameters

    # Define the headers
    headers = Dict(
        "Accept" => "application/json"
    )
    url="https://api.zbmath.org/v1/author/_structured_search?page=0&results_per_page=100&author%20name=" * HTTP.URIs.escapeuri("Demmel, J W")
    # Attempt to send the GET request
    try
        response = HTTP.get(url, headers = headers)

        if response.status == 200
            data = JSON.parse(String(response.body))
            return data
        else
            println("Request failed with status: ", response.status)
            println("Response body: ", String(response.body))
            return nothing
        end
    catch e
        println("An error occurred: ", e)
        return nothing
    end
end

# Fetch the data
author_data = fetch_author_data()

# Process the data if the request was successful
if author_data !== nothing
    for dic in author_data["result"]
        println(dic["code"])
    end
end