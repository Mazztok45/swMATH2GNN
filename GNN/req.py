import requests

def fetch_author_data():
    # Define the base URL
    base_url = "https://api.zbmath.org/v1/author/_structured_search"

    # Define potential parameter names
    possible_param_names = ["author name", "author_name", "authorName"]

    # Define the author name in different formats
    author_names = [
        "Demmel, J.W.",
        "Demmel, JW",
        "Demmel JW",
        "Demmel, J W"
    ]

    # Define other query parameters
    other_params = {
        "page": "0",
        "results_per_page": "100"
    }

    }

    for param_name in possible_param_names:
        for author_name in author_names:
            # Combine other parameters with the current author parameter
            params = {**other_params, param_name: author_name}

            print(f"Trying with parameter '{param_name}' and author name '{author_name}'")

            try:
                # Send the GET request
                response = requests.get(base_url, headers=headers, params=params)

                # Check if the request was successful
                if response.status_code == 200:
                    data = response.json()
                    print(f"Success with parameter '{param_name}' and author name '{author_name}'")
                    print(data)
                    return  # Exit after successful request
                elif response.status_code == 404:
                    print(f"No results found with parameter '{param_name}' and author name '{author_name}'.")
                else:
                    print(f"Failed with status code {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")

    print("All attempts failed. Please check the API documentation or the author name.")

if __name__ == "__main__":
    fetch_author_data()
