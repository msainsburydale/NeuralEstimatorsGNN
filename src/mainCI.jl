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
gnn = gnnarchitecture(p; propagation = "WeightedGraphConv")

# pretrain with point estimator
Flux.loadparams!(gnn, loadbestweights(path * "/runs_GNN_m$M"))

lower = deepcopy(gnn)
upper = deepcopy(gnn)
intervalestimator = IntervalEstimator(deepcopy(lower), deepcopy(upper))

α = 0.05f0
q = [α/2, 1-α/2] # quantiles
qloss = (θ̂, θ) -> quantileloss(θ̂, θ, gpu(q))

# ---- Training ----

J = 3

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

seed!(1)
K = 3000
θ, Z = variableirregularsetup(ξ, n, K = quick ? K ÷ 100 : K, m = m, neighbour_parameter = r, J = J, clustering = true)
Z = Z[1]
intervalestimator(Z)
ci = interval(intervalestimator, Z, parameter_names = ξ.parameter_names)
bool = [ci[k][i, 1] <= θ.θ[:, k][i] <= ci[k][i, 2] for i in 1:p, k in 1:size(θ, 2)]
coverage = mean(bool; dims = 2)
df = DataFrame(coverage', ξ.parameter_names)
CSV.write(path * "/marginal_coverage.csv", df)


# # ---- Conditional coverage (coverage given specific parameter value) ----
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
