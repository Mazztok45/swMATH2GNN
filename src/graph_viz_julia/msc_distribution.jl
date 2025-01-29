using DataFrames
using StatsBase
using Plots

using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe

software_df  = generate_software_dataframe()

### MSC codes
msc = unique(software_df[!,[:id, :classification]]).classification

l = reduce(vcat, [split(elem, ";") for elem in msc])

countmap(l)
plot(countmap(l))
