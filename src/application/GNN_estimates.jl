using CSV
using DataFrames
using Folds
using LinearAlgebra
using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using RData
using Statistics: mean
using StatsBase: sample

## ---- Load the data ----

model = joinpath("GP", "nuFixed")
include(joinpath(pwd(), "src", model, "model.jl"))
include(joinpath(pwd(), "src", "architecture.jl"))

path = "intermediates/application"
if !isdir(path) mkpath(path) end

## Load the clustered data as a single data frame, and then split by cluster
clustered_data = RData.load(joinpath(path, "clustered_data2.rds"))
clustered_data = [filter(:cluster => cluster -> cluster == i, clustered_data) for i in unique(clustered_data[:, :cluster])]

## Load the distance scaling factors
scale_factors = RData.load(joinpath(path, "scale_factors.rds")).data


## ---- Load estimators ----

p = 3 # number of parameters

pointestimator = gnnarchitecture(p)

a = [minimum.(values(Ω))...]
b = [maximum.(values(Ω))...]
U = gnnarchitecture(p; final_activation = identity)
V = deepcopy(U)
intervalestimator = IntervalEstimator(U, V, a, b)

Flux.loadparams!(pointestimator,    loadbestweights(joinpath(path, "pointestimator")))
Flux.loadparams!(intervalestimator, loadbestweights(joinpath(path, "intervalestimator")))

## ---- Estimate ----

function estimate(pointestimator, intervalestimator data, scale_factor)

   # Restrict the sample size for computational reasons
   n = size(data, 1)
   max_n = 1000
   if n > max_n
    data = data[sample(1:n, max_n; replace = false), :]
   end

   Z = data[:, [:Z]]         |> Matrix
   S = data[:, [:x, :y, :z]] |> Matrix

    # spatial radius
    r₀ = 0.15                # fixed radius used during training on the unit square
    r  = r₀ / scale_factor   # neighbourhood disc radius used here
    k  = 30                  # maximum number of neighbours to consider

    # Compute the adjacency matrix
    A = adjacencymatrix(S, r, k)

    # Scale the distances so that they are between [0, sqrt(2)]
    v = A.nzval
    v .*= scale_factor

    # Construct the graph
    g = GNNGraph(A, ndata = permutedims(Z))
    g = gpu(g)

    # Estimate parameters
    t = @elapsed θ = pointestimator(g)
    t += @elapsed θ_quantiles = estimator(g)
    θ = vcat(v, θ_quantiles)

    # Scale the range parameter and its quantiles back to original scale
    θ[2 .+ (0:2)p] /= scale_factor

    return θ, t
end

pointestimator    = gpu(pointestimator)
intervalestimator = gpu(intervalestimator)


total_time = @elapsed results = Folds.map(1:length(clustered_data)) do k
   estimate(pointestimator, clustered_data[k], scale_factors[k])
end
θ = broadcast(x -> x[1], results)
t = broadcast(x -> x[2], results)
θ = permutedims(hcat(θ...))
estimates = DataFrame(θ, repeat(["τ", "ρ", "σ"], 3) .* repeat(["", "_lower", "_upper"], inner = 3))
estimates[:, :time] = t
estimates[:, :total_time] .= total_time
CSV.write(joinpath(path, "GNN_estimates.csv"), estimates)

@info "Finished GNN estimation!"