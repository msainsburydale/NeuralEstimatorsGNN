using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--model"
		help = "A relative path to the folder of the assumed model; this folder should contain scripts for defining the parameter configurations in Julia and for data simulation."
		arg_type = String
		required = true
	"--skip_training"
		help = "A flag controlling whether or not we should skip training the estimators: useful for running the assessments without retraining the estimators."
		action = :store_true
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
	"--m"
		help = "The sample size to use during training. If multiple samples sizes are given as a vector, multiple neural estimators will be trained."
		arg_type = String
end
parsed_args = parse_args(arg_table)

model         = parsed_args["model"]
skip_training = parsed_args["skip_training"]
quick         = parsed_args["quick"]
m = let expr = Meta.parse(parsed_args["m"])
    @assert expr.head == :vect
    Int.(expr.args)
end

# model="GP/nuFixed"
# m=[1]
# skip_training = true
# quick=true

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using GraphNeuralNetworks
using BenchmarkTools
using DataFrames
using CSV
using NamedArrays

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
J = 3
K_train = 10_000
K_val   = K_train ÷ 10
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
end
K_test = K_val

p = ξ.p
n = ξ.n

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 1000

# ---- Estimator ----

seed!(1)
pointestimator = gnnarchitecture(p)

# pretrain with point estimator
if isfile(path * "/runs_GNN_m$M")
	Flux.loadparams!(pointestimator, loadbestweights(path * "/runs_GNN_m$M"))
end


lower = deepcopy(pointestimator)
upper = deepcopy(pointestimator)
intervalestimator = IntervalEstimator(deepcopy(lower), deepcopy(upper))

α = 0.05f0
q = [α/2, 1-α/2] # quantiles
qloss = (θ̂, θ) -> quantileloss(θ̂, θ, gpu(q))

pointestimator = PointEstimator(pointestimator)

# ---- Training ----

if !skip_training

	seed!(1)
	@info "Sampling parameter vectors used for validation..."
	θ_val = Parameters(K_val, ξ, n, J = J)
	@info "Sampling parameter vectors used for training..."
	θ_train = Parameters(K_train, ξ, n, J = J)
	@info "training the GNN..."
	trainx(intervalestimator, θ_train, θ_val, simulate, m, savepath = path * "/runs_GNN_CI", epochs = epochs, batchsize = 16, loss = qloss, epochs_per_Z_refresh = 3)

end

# ---- Load the trained estimator ----

Flux.loadparams!(intervalestimator, loadbestweights(path * "/runs_GNN_CI_m$M"))

# ---- Marginal coverage (coverage over the whole parameter space) ----

#TODO in addition to overall coverage, check that central credible intervals are
# indeed central, in the sense that alpha/2 mass lies in each tail. See Efron 2003
# "That being said, current bootstrap intervals, even nonparametric ones, are usually more accurate than their standard counterparts. “Accuracy” is a word that needs careful definition when applied to confidence intervals. The worst definition (seen unfortunately often in simulation studies of competing confidence interval techniques) concentrates on overall coverage. Even the standard intervals might come reasonably close to 90% overall coverage in the situation in Table 1, but they do so in a lopsided fashion, often failing to cover much more than 5% on the left and much less than 5% on the right. The purpose of a two- sided confidence interval is accurate inference in both directions.""
# "Coverage, even appropriately defined, is not the end of the story. Stability of the intervals, in length and location, is also important. Here is an example. Sup- pose we are in a standard normal situation where the exact interval is Student’s t with 10 degrees of free- dom. Method A produces the exact 90% interval except shortened by a factor of 0.90; method B produces the exact 90% interval either shortened by a factor of 2/3 or lengthened by a factor of 3/2, with equal proba- bility. Both methods provide about 86% coverage, but the intervals in method B will always be substantially misleading."

"""
	coverage(intervals::V, θ) where  {V <: AbstractArray{M}} where M <: AbstractMatrix

Given a p×K matrix of true parameters `θ`, determine the empirical coverage of
a collection of confidence `intervals` (a K-vector of px2 matrices).

The overall empirical coverage is obtained by averaging the resulting 0-1 matrix
elementwise over all parameter vectors.

# Examples
```
p = 3
K = 100
θ = rand(p, K)
intervals = [rand(p, 2) for _ in 1:K]
coverage(intervals, θ)
```
"""
function coverage(intervals::V, θ) where  {V <: AbstractArray{M}} where M <: AbstractMatrix

    p, K = size(θ)
	if K == 1
		K = length(intervals)
		θ = repeat(θ, 1, K)
	end
	@assert length(intervals) == K
	@assert all(size.(intervals, 1) .== p)
	@assert all(size.(intervals, 2) .== 2)

	# for each confidence interval, determine if the true parameters, θ, are
	# within the interval.
	within = [intervals[k][i, 1] <= θ[i, k] <= intervals[k][i, 2] for i in 1:p, k in 1:K]

	# compute the empirical coverage
	cvg = mean(within, dims = 2)

	return cvg
end

# Test with respect to a set of irregular uniformly sampled locations
seed!(1)
S = rand(n, 2)

# Simulate data
seed!(1)
K = quick ? 100 : 3000
θ = Parameters(K, ξ, n, J = 1)
Z = simulate(θ, m)

# Marginal coverage for
intervals = interval(intervalestimator, Z, parameter_names = ξ.parameter_names)
cvg = coverage(intervals, θ.θ)
df  = DataFrame(cvg', ξ.parameter_names)
CSV.write(path * "/marginal_coverage.csv", df)


# ---- Conditional coverage (coverage given specific parameter values) ----

#TODO add plots of the conditonal coverage in the supplementary material at the revision stage

#TODO m should be an argument of this function, as should simulator
# """
# 	conditonalcoverage(estimator::Union{PointEstimator, IntervalEstimator}, θ, S, ξ; J::Integer = 1000)
#
# Computes the empirical coverage conditional on a single parameter configuration `θ` (stored as a
# p×1 matrix) over the spatial sample locations `S`, based on `J` simulations from the model.
#
# The credible intervals are consructed using either a parametric bootstrap (if
# `estimator` is a `PointEstimator`) or via the marginal posterior quantiles
# (if `estimator` is an `IntervalEstimator`).
# """
# function conditonalcoverage(estimator, θ, S, ξ; J::Integer = 1000)
#
# 	typeof(estimator) <: IntervalEstimator ? method = "quantile" : method = "bootstrap"
#
# 	p, K = size(θ)
# 	@assert K == 1
#
# 	D = pairwise(Euclidean(), S, S, dims = 1)
# 	A = adjacencymatrix(D, ξ.δ)
# 	g = GNNGraph(A)
# 	ξ = (ξ..., S = S, D = D)      # update ξ to contain the new distance matrix D (needed for simulation and ML estimation)
# 	parameters = Parameters(θ, ξ)
#
# 	# simulate a large number J of data sets
# 	Z = simulate(parameters, M, J)
# 	Z = reshapedataGNN(Z, g)
#
# 	if method == "quantile"
# 		# Compute coverage using the posterior quantile estimator
# 		intervals = interval(estimator, Z) # TODO need to document this method of interval in NeuralEstimators
# 		cvg = coverage(intervals, θ)
# 	else
# 		# Compute coverage using parametric bootstrap
# 		B = 400
# 		intervals = map(Z) do z
# 			θ̂ = estimator(z)
# 			params = Parameters(θ̂, ξ)
# 			Z̃ = simulate(params, M, B)
# 			Z̃ = reshapedataGNN(Z̃, g)
# 			# θ̃ = estimateinbatches(pointestimator, Z̃; batchsize = 400)
# 			θ̃ = pointestimator(Z̃)
# 			interval(θ̃; probs = [0.025, 0.975])
# 		end
# 		cvg = coverage(intervals, θ)
# 	end
#
# 	# store the coverage values, true parameters, and method name in a dataframe
# 	df = DataFrame(
# 		method = repeat([method], p),
# 		parameter = ξ.parameter_names,
# 		parameter_value = vec(θ),
# 		coverage = vec(cvg)
# 		)
#
#
# 	return df
# end
#
# import NeuralEstimatorsGNN: Parameters
# function Parameters(θ::Matrix, ξ)
#
# 	p, K = size(θ)
#
# 	# Determine if we are estimating ν and σ
# 	parameter_names = String.(collect(keys(ξ.Ω)))
# 	estimate_ν = "ν" ∈ parameter_names
# 	estimate_σ = "σ" ∈ parameter_names
#
# 	# GP covariance parameters and Cholesky factors
# 	ρ_idx = findfirst(parameter_names .== "ρ")
# 	ν_idx = findfirst(parameter_names .== "ν")
# 	σ_idx = findfirst(parameter_names .== "σ")
# 	ρ = θ[ρ_idx, :]
# 	ν = estimate_ν ? θ[ν_idx, :] : fill(ξ.ν, K)
# 	σ = estimate_σ ? θ[σ_idx, :] : fill(ξ.σ, K)
# 	chols = maternchols(ξ.D, ρ, ν, σ.^2; stack = false)
# 	chol_pointer = collect(1:K)
#
# 	# Construct the graphs
# 	A = adjacencymatrix.(ξ.D, ξ.δ)
# 	graphs = GNNGraph.(A)
#
# 	Parameters(θ, ξ.S, graphs, chols, chol_pointer)
# end
#
# # Do this by constructing a grid of parameters, and testing the coverage at
# # each parameter value. This way, we can produce a coverage map.
#
# # Support that we want to test over
# # convert to array since broadcasting over dictionaries and NamedTuples is reserved
# # "narrow" the prior support to avoid boundary effects
# test_support = map([Ω...]) do x
# 	[minimum(x) * 1.1, maximum(x) * 0.9]
# end
#
# # Grid of parameter values
# l = 10
# x = range(test_support[1][1], test_support[1][2], length = l)
# y = range(test_support[2][1], test_support[2][2], length = l)
# grd = expandgrid(x, y)

# Estimate the coverage conditional on parameter configuration
# for set ∈ ["uniform", "quadrants", "mixedsparsity", "cup"]
# 	seed!(1)
# 	S = spatialconfigurations(n, set)
# 	dfs = map(1:size(grd, 1)) do k
# 		θ = grd[k, :, :]
# 		conditonalcoverage(intervalestimator, θ, S, ξ, J = 1000)
# 	end
# 	df = vcat(dfs...)
# 	df[:, :set] = repeat([set], nrow(df)) # add set information
# 	CSV.write(path * "/conditionalcoverage_$set.csv", df)
# end
