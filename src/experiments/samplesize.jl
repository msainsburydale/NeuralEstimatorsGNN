using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--model"
		help = "A relative path to the folder of the assumed model; this folder should contain scripts for defining the parameter configurations in Julia and for data simulation."
		arg_type = String
		required = true
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
model       = parsed_args["model"]
quick       = parsed_args["quick"]


# ----

using NeuralEstimators
using NeuralEstimatorsGNN
using DataFrames
using Distances: pairwise, Euclidean
using GraphNeuralNetworks
using CSV
using DelimitedFiles
using NamedArrays

# TODO For a fixed spatial domain, the average distance between spatial locations
#   decreases as the sample size increases; therefore, GNN-based estimators for
#   spatial problems may be more robust to different sample sizes than those
#   used during training if neighbours are defined by some fixed spatial radius.
# I should note that I am defining neighbours based on spatial proximity rather than a K-nearest neighbours approach: that is, I define the neighbours of a node v as all points u that fall within some fixed spatial distance of v. This adds some robustness to the estimator when dealing with varying sample sizes n when compared with simply choosing the K-nearest neighbours since, with a fixed spatial domain, the average distance between spatial locations decreases as the sample size increases and, hence, the K-nearest neighbours of a node will asymptotically, which may compromise the estimation of range parameters.

# neighbours = 8 # number of neighbours to consider
ϵ = 0.05f0 # spatial radius within which nodes are neighbours

model = "GaussianProcess/fourparameters" #TODO change to allow model as an argument; make this experiment consistent with the other experiments
m = 1 # number of replicates per spatial field
n = 512 # number of observations per field during training

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 20000
K_val   = K_train ÷ 10
K_test = 1000
if quick
	K_train = K_train ÷ 100
	K_val   = K_val ÷ 100
	K_test = K_test ÷ 100
end


# ---- Training ----

# Construct a variable set of irregular locations {Sₖ : k = 1, …, K}
#TODO should do on-the-fly simulation to improve results. This may be possible
# with the fast simulation techniques.
θ_val,   Z_val    = variableirregularsetup(ξ, K = K_val, n = n, J = 5)
θ_train, Z_train  = variableirregularsetup(ξ, K = K_train, n = n, J = 5)

# Train a point estimator for the posterior median
seed!(1)
pointestimator = gnnarchitecture(ξ.p; globalpool = "deepset")
train(pointestimator, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_point")
Flux.loadparams!(pointestimator, loadbestweights(path * "/runs_point"))

# Confidence-interval estimator based on the posterior quantiles
l = deepcopy(pointestimator)
u = deepcopy(pointestimator)
ciestimator = IntervalEstimator(l, u)
train(ciestimator, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_ci", loss = (θ̂, θ) -> quantileloss(θ̂, θ, gpu([0.025, 0.975])))
Flux.loadparams!(ciestimator, loadbestweights(path * "/runs_ci"))
ciestimator = ciestimator |> gpu

# ---- Assessment: training n ----

# Construct the test set
seed!(1)
θ, Z = variableirregularsetup(ξ, K = K_test, n = n)

# Construct and assess confidence intervals
ci = interval(ciestimator, Z; parameter_names = parameter_names)
assessci(ci, θ.θ, "n$n")

# ---- Assessment: larger n ----

# Construct the test set
seed!(1)
largern = 4n
θ, Z = variableirregularsetup(ξ, K = K_test, n = largern)

# Construct and assess confidence intervals
ci = interval(ciestimator, Z; parameter_names = parameter_names)
assessci(ci, θ.θ, "n$largern")


# ---- Apply the estimator to the massive spatial data sets ----

#TODO which data sets do I need to estimate from (competitions 1a and 1b?)

datapath = "data/kaust" # relative path to the kaust data
internal = true # set to false to do inference over the competition data
thinned  = true # set to false to use the full data sets
competition = "a" # can also set to "b"

datasets = [1, 2, 4]

cis = map(datasets) do i

	@info "Estimating over data set $i"

	# load the data
	datpth = datapath
	datpth *= internal ?  "_internal" : ""
	datpth *= thinned ?  "/thinned" : "/full"
	loadpath = joinpath(pwd(), datpth, "Sub-competition_1$competition")
	data = readdlm(loadpath * "/Train_$i.csv",  ',')

	# split the data into spatial locations and response values
	S = data[:, 1:2]
	Z = data[:, 3]
	Z = reshape(Z, length(Z), 1) #TODO shouldn't have to do this
	S = Float64.(S)
	Z = Float64.(Z)

	# prepare the data for the GNN
	A = adjacencymatrix(S, ϵ)
	g = GNNGraph(A)
	Z = reshapedataGNN([Z], g)

	# confidence interval
	ci = interval(ciestimator, Z; parameter_names = parameter_names)[1]
	savenamedarray(ci, loadpath * "/ci_$i.csv")
	ci
end




# ---- unused code ----


# # Construct a set of randomly sampled irregular locations, S
# seed!(1)
# S = rand(n, 2)
# D = pairwise(Euclidean(), S, S, dims = 1)
# A = adjacencymatrix(D, ϵ)
# g = GNNGraph(A)
# ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
# θ = Parameters(ξ, K_test)
# Z = simulate(θ, m)
# Z = reshapedataGNN(Z, g)
