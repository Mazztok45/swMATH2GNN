import Pkg

# Get the current directory and define environment names
current_dir = pwd()
project_name = "swMATH2GNN"
clean_env_name = "clean_env"

if basename(current_dir) == project_name
    # Construct the full path to the clean_env directory
    target_dir = joinpath(current_dir, clean_env_name)
    if isdir(target_dir)
        println("Switching to clean_env at $target_dir...")
        cd(target_dir)
        
        # Activate the environment
        Pkg.activate(".")
        
        # Add packages to the environment
        packages = [
            Pkg.PackageSpec(name="JSON3"),
            Pkg.PackageSpec(name="DataFrames"),
            Pkg.PackageSpec(name="Arrow"),
            Pkg.PackageSpec(name="SparseArrays"),
            Pkg.PackageSpec(name="MLLabelUtils"),
            Pkg.PackageSpec(name="StatsBase", version="0.33.21"),
            Pkg.PackageSpec(name="Serialization"),
            Pkg.PackageSpec(name="Random"),
            Pkg.PackageSpec(name="Statistics"),
            Pkg.PackageSpec(name="Flux"),
            Pkg.PackageSpec(name="GraphNeuralNetworks", version="0.6.7"),
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

    else
        println("Target directory '$clean_env_name' does not exist!")
    end
else
    println("Not in the '$project_name' directory. No action taken.")
end
