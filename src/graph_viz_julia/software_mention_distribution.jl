########### graph on software mention distribution amongs papers
using Plots
counts = Dict{String, Int}()

# Iterate over the nested arrays and count each string
for arr in software_arrays
    for value in arr
        counts[value] = get(counts, value, 0) + 1
    end
end


# Prepare data for plotting
string_counts = collect(values(counts))  # List of counts only

# Define y-tick positions (from 0 to 28000 with a step of 5000)
ytick_positions = 0:5000:28000

# Plotting the histogram with custom y-ticks
hist_plot=histogram(
    string_counts, 
    bins=10, 
    title="Histogram of Software Occurrences", 
    xlabel="Software Occurrences Count", 
    ylabel="Frequency", 
    yticks=(ytick_positions, string.(ytick_positions)),# An example of semi-supervised node classification
    label=false   # Use the defined y-ticks with integer labels
)
# Save the plot to a file (e.g., PNG)
savefig(hist_plot, "software_mentions_distribution.png")
########### 