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
r = ξ.r

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 1000

# ---- Estimator ----

seed!(1)
gnn = gnnarchitecture(p)

# pretrain with point estimator
if isfile(path * "/runs_GNN_m$M")
	Flux.loadparams!(gnn, loadbestweights(path * "/runs_GNN_m$M"))
end


lower = deepcopy(gnn)
upper = deepcopy(gnn)
intervalestimator = IntervalEstimator(deepcopy(lower), deepcopy(upper))

α = 0.05f0
q = [α/2, 1-α/2] # quantiles
qloss = (θ̂, θ) -> quantileloss(θ̂, θ, gpu(q))

# ---- Training ----

if !skip_training

	seed!(1)
	@info "simulating training data for the GNN..."
	θ_val,   Z_val   = variableirregularsetup(ξ, n, K = K_val, m = m, neighbour_parameter = r, J = J, clustering = true)
	θ_train, Z_train = variableirregularsetup(ξ, n, K = K_train, m = m, neighbour_parameter = r, J = J, clustering = true)
	@info "training the GNN-based credible-interval estimator..."
	trainx(intervalestimator, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN_CI", epochs = epochs, batchsize = 16, loss = qloss)

end

# ---- Load the trained estimator ----

Flux.loadparams!(intervalestimator, loadbestweights(path * "/runs_GNN_CI_m$M"))

# ---- Marginal coverage (coverage over the whole parameter space) ----

#TODO in addition to overall coverage, check that central credible intervals are
# indeed central, in the sense that alpha/2 mass lies in each tail. 

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

# TODO for both things that I want to do, I need a constructor that takes in
# a matrix of parameters. Start by cleaning up NeuralEstimatorsGNN.jl, and
# then figuring out how to add this constructor.


# Simulate data
seed!(1)
K = quick ? 100 : 3000
θ, Z, ξ = variableirregularsetup(ξ, n, K = K, m = m, neighbour_parameter = r, J = J, clustering = true, return_ξ = true)
Z = Z[1]

# Maringal coverage for
intervals = interval(intervalestimator, Z, parameter_names = ξ.parameter_names)
cvg = coverage(intervals, θ.θ)
df  = DataFrame(cvg', ξ.parameter_names)
CSV.write(path * "/marginal_coverage.csv", df)


# TODO also compute coverage based on parametric bootstrap. Note that parametric
# bootstrap is not actually that inefficient, since we just need to do one
# Cholesky factorisation for each estimate.
gnn(Z)


# # ---- TODO Conditional coverage (coverage given specific parameter value) ----
#
# # Do this by constructing a grid of parameters, and testing the coverage at each
# # parameter value. This way, we can produce a coverage map.
#
# # Support that we want to test over
# # convert to array since broadcasting over dictionaries and NamedTuples is reserved
# # "narrow" the prior support to avoid boundary effects
# test_support = map([Ω...]) do x
# 	[minimum(x) * 1.1, maximum(x) * 0.9]
# end
#
# # 25 * 25 grid of parameter values
# x = range(test_support[1][1], test_support[1][2], length = 25)
# y = range(test_support[2][1], test_support[2][2], length = 25)
#
# grd = expandgrid(x, y)
#
# # estimate the coverage for each parameter configuration
# k = 1 # k ∈ 1:size(grd, 1)
# Parameters(grd[k, :],
#
#
# # save the grid of parameters and the corresponding coverage
# grd
