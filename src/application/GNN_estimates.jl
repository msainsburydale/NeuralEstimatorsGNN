using CSV
using CUDA
using DataFrames
using Folds
using LinearAlgebra
using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using RData
using Statistics: mean
using StatsBase: sample
using SparseArrays


## ---- Load the data ----

model = joinpath("GP", "nuFixed")
include(joinpath(pwd(), "src", model, "model.jl"))
include(joinpath(pwd(), "src", "architecture.jl"))

## Load the clustered data as a single data frame, and then split by cluster
path = "intermediates/application"
clustered_data = RData.load(joinpath(path, "clustered_data2.rds"))
clustered_data = [filter(:cluster => cluster -> cluster == i, clustered_data) for i in unique(clustered_data[:, :cluster])]

## Load the distance scaling factors
scale_factors = RData.load(joinpath(path, "scale_factors.rds")).data

## ---- Load estimators ----

p = 3 # number of parameters

pointestimator = gnnarchitecture(p)

v = gnnarchitecture(p; final_activation = identity)
a = [minimum.(values(Ω))...]
b = [maximum.(values(Ω))...]
g = Compress(a, b)
intervalestimator = IntervalEstimator(v, g)

loadpath  = joinpath(path, "pointestimator", "best_network.bson")
@load loadpath model_state
Flux.loadmodel!(pointestimator, model_state)

loadpath  = joinpath(path, "intervalestimator", "best_network.bson")
@load loadpath model_state
Flux.loadmodel!(intervalestimator, model_state)


## ---- Estimate ----

@info "Starting GNN estimation..."

function constructgraph(data, scale_factor)

    # Restrict the sample size while prototyping
    #n = size(data, 1)
    #max_n = 2000
    #if n > max_n
    #  data = data[sample(1:n, max_n; replace = false), :]
    #end

    # Compute the adjacency matrix
    S = data[:, [:x, :y, :z]] |> Matrix
    S = Float32.(S)
    r = 0.15 # disc radius on the unit square
    k = 30   # maximum number of neighbours to consider 
    A = adjacencymatrix(S, r / scale_factor, k)

    # Scale the distances so that they are between [0, sqrt(2)]
    v = A.nzval
    v .*= scale_factor

    # Construct the graph
    Z = data[:, [:Z]] |> Matrix
    Z = Float32.(Z)
    GNNGraph(A, ndata = (Z = permutedims(Z), ))
end

t = @elapsed g = Folds.map(1:length(clustered_data)) do k
   constructgraph(clustered_data[k], scale_factors[k])
end
t += @elapsed θ = estimateinbatches(pointestimator, g)
t += @elapsed θ_quantiles = estimateinbatches(intervalestimator, g)
θ = vcat(θ, θ_quantiles)

# Scale the range parameter point and quantile estimates back to original scale
for k in 1:size(θ, 2)
  θ[2 .+ (0:2)p, k] /= scale_factors[k]
end

θ = permutedims(θ)
θ = DataFrame(θ, repeat(["τ", "ρ", "σ"], 3) .* repeat(["", "_lower", "_upper"], inner = 3)) 
CSV.write(joinpath(path, "GNN_runtime.csv"), DataFrame(time = [t]))
CSV.write(joinpath(path, "GNN_estimates.csv"), θ)

@info "Finished GNN estimation!"
