using CSV
using DataFrames
using Folds
using LinearAlgebra
using NeuralEstimators
using Optim
using RData
using Statistics: mean
using StatsBase: sample

## ---- Load the data ----

model = joinpath("GP", "nuFixed")
include(joinpath(pwd(), "src", model, "model.jl"))

path = "intermediates/application"
if !isdir(path) mkpath(path) end

## Load the clustered data as a single data frame, and then split by cluster
clustered_data = RData.load(joinpath(path, "clustered_data2.rds"))
clustered_data = [filter(:cluster => cluster -> cluster == i, clustered_data) for i in unique(clustered_data[:, :cluster])]

## Load the distance scaling factors
scale_factors = RData.load(joinpath(path, "scale_factors.rds")).data

## ---- MLE functions ----

## Negative log-likelihood function to be minimised using Optim
function nll(θ, Z, D, prior = nothing)

	# Constrain the estimates to be valid
  if isnothing(prior)
 	  θ = exp.(θ)
 	else
	  θ = scaledlogistic.(θ, prior)
  end

  # Extract parameters
	τ = θ[1]
	ρ = θ[2]
	σ = θ[3]
	σ² = σ^2
	ν = one(eltype(θ)) # smoothness fixed to 1

	# Covariance matrix (exploit symmetry to minimise number of computations)
  Σ = matern.(UpperTriangular(D), ρ, ν, σ²)
	Σ[diagind(Σ)] .+= τ^2

	# Log-likelihood function
	ℓ = gaussiandensity(Z, Σ; logdensity = true)

	return -ℓ
end

"""
scale_factor: a single number used to scale the distances to [0, √2]
prior: a p-dimensional vector of 2-dimensional vectors specifying the lower and upper bound for each parameter
"""
function ML(Z, S, θ₀; scale_factor = nothing, prior = nothing)

	# Convert to Float64 to avoid numerical errors
	Z  = broadcast.(Float64, Z)
	θ₀ = Float64.(θ₀)

	# Compute the distance matrix
	D = pairwise(Euclidean(), S, dims = 1) 
	if !isnothing(scale_factor) D .*= scale_factor end

	## Estimate the parameters by minimising the negative log-likelihood
 	if isnothing(prior)
	  θ₀ = log.(θ₀)  # log initial values since we exponentiate during optimisation
	  θ  = optimize(θ -> nll(θ, Z, D), θ₀, NelderMead()) |> Optim.minimizer
	  θ  = exp.(θ)
	else
    θ₀ = scaledlogit.(θ₀, prior)
    θ  = optimize(θ -> nll(θ, Z, D, prior), θ₀, NelderMead()) |> Optim.minimizer
    θ  = scaledlogistic.(θ, prior)
  end

  if !isnothing(scale_factor) θ[2] /= scale_factor end

	return θ
end

#	## Estimate the parameters by minimising the negative log-likelihood
# 	if isnothing(prior)
# 	  loss(θ) = nll(θ, Z, D) # closure that will be minimised
	 #θ₀ = log.(θ₀)  # log initial values since we exponentiate during optimisation
	 # θ̂  = optimize(loss, θ₀, NelderMead()) |> Optim.minimizer
	  #θ̂  = exp.(θ̂)   # exponentiate from the log-scale
	#else
	 # loss(θ) = nll(θ, Z, D, prior) # closure that will be minimised
    #θ₀ = scaledlogit.(θ₀, prior)
    #θ̂  = optimize(loss, θ₀, NelderMead()) |> Optim.minimizer
    #θ̂  = scaledlogistic.(θ̂, prior)
  #end

# ---- MLE over clusters in parallel ----

θ₀ = [0.5, 0.3, 1.5]
prior = extrema.([Ω...])

total_time = @elapsed results = Folds.map(1:length(clustered_data)) do k

   data = clustered_data[k]
   n = size(data, 1)

   # Restrict the sample size for computational reasons
   # max_n = 4000
   max_n = 2000
   if n > max_n
    data = data[sample(1:n, max_n; replace = false), :]
   end

   Z = data[:, [:Z]]         |> Matrix
   S = data[:, [:x, :y, :z]] |> Matrix

   # Given the scale of this estimation task, it is critical that this doesn't
   # stop because of one or two failed Cholesky factorisations
   try
    t = @elapsed  θ̂ = ML(Z, S, θ₀; scale_factor = scale_factors[k], prior = prior)
		return θ̂, t
   catch
	 	@warn "Cholesky factorisation failed for cell cluster $k"
		θ̂ = [missing, missing, missing]
		t = missing
		return θ̂, t
   end
end
θ̂ = broadcast(x -> x[1], results)
t = broadcast(x -> x[2], results)
θ̂ = permutedims(hcat(θ̂...))
estimates = DataFrame(θ̂, [:τ, :ρ, :σ])
estimates[:, :time] = t
estimates[:, :total_time] .= total_time
CSV.write(joinpath(path, "ML_estimates.csv"), estimates)

@info "Finished maximum-likelihood estimation!"
