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
	K_train = K_train ÷ 10
	K_val   = K_val   ÷ 10
end
K_test = K_val

p = ξ.p
n = ξ.n

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 20 : 200

# ---- Estimator ----

seed!(1)
pointestimator = gnnarchitecture(p)
U = gnnarchitecture(p; final_activation = identity)
V = gnnarchitecture(p; final_activation = identity)
Ω = ξ.Ω
a = [minimum.(values(Ω))...]
b = [maximum.(values(Ω))...]
c = Compress(a, b)
intervalestimator = IntervalEstimator(U, V, c)

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
	trainx(intervalestimator, θ_train, θ_val, simulate, m, savepath = path * "/runs_GNN_CI", epochs = epochs, batchsize = 16, loss = qloss, epochs_per_Z_refresh = 3, stopping_epochs = 3)
end

# ---- Load the trained estimator ----

Flux.loadparams!(intervalestimator, loadbestweights(path * "/runs_GNN_CI_m$M"))

# ---- Marginal coverage (coverage over the whole parameter space) ----

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
Z = simulate(θ, M)

# Marginal coverage
intervals = interval(intervalestimator, Z, parameter_names = ξ.parameter_names)
cvg = coverage(intervals, θ.θ)
df  = DataFrame(cvg', ξ.parameter_names)
CSV.write(path * "/marginal_coverage_interval-estimator.csv", df)
