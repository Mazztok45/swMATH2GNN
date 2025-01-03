import Pkg

# Define a list of packages with optional version specifications
packages = [
    Pkg.PackageSpec(name="JSON3"),
    Pkg.PackageSpec(name="DataFrames"),
    Pkg.PackageSpec(name="Arrow"),
    Pkg.PackageSpec(name="SparseArrays"),
    Pkg.PackageSpec(name="MLLabelUtils"),
    Pkg.PackageSpec(name="StatsBase", version="0.33.21"),  # Specify version 0.33.21 for StatsBase
    Pkg.PackageSpec(name="Serialization"),
    Pkg.PackageSpec(name="Random"),
    Pkg.PackageSpec(name="Statistics"),
    Pkg.PackageSpec(name="Flux"),
    Pkg.PackageSpec(name="GraphNeuralNetworks", version="1.0.0"),  # Specify version 1.0.0 for GraphNeuralNetworks
    Pkg.PackageSpec(name="Graphs"),
    Pkg.PackageSpec(name="MultivariateStats"),
    Pkg.PackageSpec(name="StructTypes"),
    Pkg.PackageSpec(name="JLD2"),
    Pkg.PackageSpec(name="Metis"),
    Pkg.PackageSpec(name="BSON"),
    Pkg.PackageSpec(name="LinearAlgebra"),
    Pkg.PackageSpec(name="IterativeSolvers"),
    Pkg.PackageSpec(name="HTTP"),
    Pkg.PackageSpec(name="JSON"),
    Pkg.PackageSpec(name="MLUtils"),
    Pkg.PackageSpec(name="Plots"),
    Pkg.PackageSpec(name="GraphPlot"),
    Pkg.PackageSpec(name="Compose"),
    Pkg.PackageSpec(name="Cairo"),
    Pkg.PackageSpec(name="Fontconfig"),
    Pkg.PackageSpec(name="TextAnalysis"),
    Pkg.PackageSpec(name="LightGraphs"),
    Pkg.PackageSpec(name="Transformers"),
    Pkg.PackageSpec(name="Distances"),
    Pkg.PackageSpec(name="NPZ"),
    Pkg.PackageSpec(name="NearestNeighbors"),
    Pkg.PackageSpec(name="LIBSVM"),
    Pkg.PackageSpec(name="CSV")
]

# Add all packages in a single command
Pkg.add(packages)
