if basename(pwd()) != "swMATH2GNN"
    cd("..")
end 

using DataFrames
using StatsBase
using Plots
include("../extracting_data/extract_software_metadata.jl")
#using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe

software_df  = generate_software_dataframe()

#Graph

dfs=unique(software_df[:,[:id,:classification]])
# Create a new column for classifications as an array of strings
dfs.classification_split = [split(x, ";") for x in dfs.classification]
 # Flatten the list of classifications and convert it into a DataFrame
 classification_list = vcat(dfs.classification_split...)
 # Flatten the list of classifications and convert it into a DataFrame
 classification_counts = DataFrame(classification=classification_list)
# Count the occurrences of each classification
classification_summary = combine(groupby(classification_counts, :classification), nrow => :count)


# Sort the DataFrame by classification in ascending order

sorted_classification_summary = sort(classification_summary, :classification)

# Extract data from the sorted DataFrame
classifications = sorted_classification_summary.classification
counts = sorted_classification_summary.count

# Determine appropriate y-tick values
yticks_values = 0:2000:maximum(counts)

bar_plot = bar(classifications, counts, 
    xlabel = "MSC Code", 
    ylabel = "Software count", 
    title = "Number of Software Projects Categorized by MSC Code", 
    legend = false, 
    xtickfont = font(8), 
    rotation = 45,
    yticks = (yticks_values, [string(v) for v in yticks_values]),
    size = (900, 600))  # Increase plot size to fit title

# Display the plot

# Save the plot to a file (e.g., PNG)
savefig(bar_plot, "msc_code_distribution.png")

# You can also use other formats like:
# savefig(bar_plot, "msc_code_distribution.pdf")
# savefig(bar_plot, "msc_code_distribution.svg")