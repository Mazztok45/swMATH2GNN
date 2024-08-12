using DataFrames
using Graphs
using GraphPlot
using Compose
using StatsBase
using Plots
include("../GNN_Julia/hetero_data.jl")
include("../extracting_data/extract_software_metadata.jl")
using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe
import Cairo
import Fontconfig

software_df  = generate_software_dataframe()

filtered_df = software_df
related_soft_filtered_df = [collect(row)[1][2] for row in filtered_df.related_software]
g_viz=DiGraph()
add_vertices!(g_viz, length(Set(filtered_df.id)))
for i in 1:nrow(filtered_df)
    add_edge!(g_viz,filtered_df[i, :id],related_soft_filtered_df[i])
end

##### VIZ PART for swMATH software 825###
sg = induced_subgraph(g_viz, neighbors(g_viz,825))
vertices_l = reduce(hcat, [[src(e), dst(e)] for e in edges(sg[1])])
g_viz2=DiGraph()

add_vertices!(g_viz2, length(Set(vertices_l)))
for edge in edges(sg[1])
    add_edge!(g_viz2, src(edge), dst(edge))
end

l_s = unique(software_df[!,[:name,:id]])
nodelabel = [l_s[l_s.id  .== id_s,:].name[1] for id_s in sg[2]]
nodesize = fill(5, 20)

draw(PNG("soft_825_graph.png", 50cm, 50cm), gplot(g_viz2, nodelabel=nodelabel, nodesize=nodesize))

### MSC codes
msc = unique(software_df[!,[:id, :classification]]).classification

l = reduce(vcat, [split(elem, ";") for elem in msc])

countmap(l)
plot(countmap(l))
