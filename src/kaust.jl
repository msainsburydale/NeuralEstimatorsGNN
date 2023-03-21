# Strategy:
# - Train a neural credible-interval estimator for n = 512 using randomly sampled
#   irregular spatial locations.
# - Assess the neural credible-interval estimator for n ∈ {512, 2048}. This will
#   indicate whether the estimator is performing reasonably well for sample sizes
#   used during training, and whether there is a substantial loss of efficiency
#   when extrapolating to different sample sizes.


# Possible things that can help:
# - Develop code for fast simulation of Gaussian processes that does not require
#   the Cholesky factor (see GaussianRandomFields.jl).


using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
quick       = parsed_args["quick"]


# ---- Some functions taken from Neural-confidence/src/experiment.jl and elsewhere ----

function savenamedarray(a, savepath)
	df = DataFrame(a, Symbol.(names(a, 2)))
	insertcols!(df, 1, Symbol(dimnames(a, 1)) => names(a, 1))
	CSV.write(savepath, df)
end

function empiricalintervalscore(ci, θ, α; per_parameter::Bool = true)
	p, K = size(θ)
	@assert length(ci) == K
	@assert all(size.(ci, 1) .== p)
	@assert all(size.(ci, 2) .== 2)

	is = map(1:K) do k
		l = ci[k][:, 1]
		u = ci[k][:, 2]
		t = θ[:, k]

		if !per_parameter
			intervalscore(l, u, t, α)
		else
			map(1:p) do i
				intervalscore(l[i], u[i], t[i], α)
			end
		end
	end
	mean(is)
end

function assessci(ci::V, θ, name::String) where  {V <: AbstractArray{M}} where M <: AbstractMatrix
	cvg = vec(coverage(ci, θ))
	is  = empiricalintervalscore(ci, θ, 0.05)
	df  = vcat(cvg', is')
	score_names = ["coverage", "IS"]
	df = NamedArray(df, (score_names, parameter_names))
	savenamedarray(df, path * "/scores_$name.csv")
end

function variableirregularsetup(ξ; K, n, J = 1, m = 1)

	D = map(1:K) do k
		S = rand(n, 2)
		D = pairwise(Euclidean(), S, S, dims = 1)
		D
	end
	A = adjacencymatrix.(D, ϵ)
	g = GNNGraph.(A)

	ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
	θ = Parameters(ξ, K, J = J)
	Z = simulate(θ, m)

	g = repeat(g, inner = J)
	Z = reshapedataGNN(Z, g)

	return θ, Z
end


# ----

using NeuralEstimators
using NeuralEstimatorsGNN
using DataFrames
using Distances: pairwise, Euclidean
using GraphNeuralNetworks
using CSV
using DelimitedFiles
using NamedArrays

# NB For a fixed spatial domain, the average distance between spatial locations
#   decreases as the sample size increases; therefore, GNN-based estimators for
#   spatial problems may be more robust to different sample sizes than those
#   used during training if neighbours are defined by some fixed spatial radius.

# neighbours = 8 # number of neighbours to consider
ϵ = 0.05f0 # spatial radius within which nodes are neighbours

model = "GaussianProcess/fourparameters"
m = 1 # number of replicates per spatial field
n = 512 # number of observations per field during training


include(joinpath(pwd(), "src/$model/Parameters.jl"))
include(joinpath(pwd(), "src/$model/Simulation.jl"))
include(joinpath(pwd(), "src/Architecture.jl"))

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
ciestimator = CIEstimator(l, u)
train(ciestimator, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_ci", loss = quantileloss(θ̂, θ, gpu([0.025, 0.975])))
Flux.loadparams!(ciestimator, loadbestweights(path * "/runs_ci"))
ciestimator = ciestimator |> gpu

# ---- Assessment: training n ----

# Construct the test set
seed!(1)
θ, Z = variableirregularsetup(ξ, K = K_test, n = n)

# Construct and assess confidence intervals
ci = confidenceinterval(ciestimator, Z; parameter_names = parameter_names)
assessci(ci, θ.θ, "n$n")

# ---- Assessment: larger n ----

# Construct the test set
seed!(1)
largern = 2048
θ, Z = variableirregularsetup(ξ, K = K_test, n = largern)

# Construct and assess confidence intervals
ci = confidenceinterval(ciestimator, Z; parameter_names = parameter_names)
assessci(ci, θ.θ, "n$largern")


# ---- Apply the estimator to the massive spatial data sets ----

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
	ci = confidenceinterval(ciestimator, Z; parameter_names = parameter_names)[1]

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
