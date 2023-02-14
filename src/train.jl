using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--model"
		help = "A relative path to the folder of the assumed model; this folder should contain scripts for defining the parameter configurations in Julia and for data simulation."
		arg_type = String
		required = true
	"--n"
		help = "The number of observations in a single field."
		arg_type = Int
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
model       = parsed_args["model"]
n           = parsed_args["n"]
quick       = parsed_args["quick"]


using NeuralEstimators
using NeuralEstimatorsGNN
using Distances: pairwise, Euclidean
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/Parameters.jl"))
include(joinpath(pwd(), "src/$model/Simulation.jl"))
include(joinpath(pwd(), "src/Architecture.jl"))

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10000
K_val   = K_train ÷ 5
K_test  = K_val
if quick
	K_train = K_train ÷ 100
	K_val   = K_val ÷ 100
	K_test  = K_test ÷ 100
end

p = ξ.p
ϵ = 0.125f0 # distance cutoff used to define the local neighbourhood of each node


function variableirregularsetup(ξ; K, n, J = 10)

	D = map(1:K) do k
		S = rand(n, 2)
		D = pairwise(Euclidean(), S, S, dims = 1)
		D
	end
	A = adjacencymatrix.(D, ϵ = ϵ)
	g = GNNGraph.(A)

	ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
	θ = Parameters(ξ, K, J = J)
	m = 1 # number of replicates per spatial field
	Z = simulate(θ, m)

	g = repeat(g, inner = J)
	Z = reshapedataGNN(Z, g)

	return θ, Z
end

# Construct a variable set of irregular locations {Sₖ : k = 1, …, K}
θ_val,   Z_val    = variableirregularsetup(ξ, K = K_val, n = n)
θ_train, Z_train  = variableirregularsetup(ξ, K = K_train, n = n)

# GNN estimator
seed!(1)
GNN = gnnarchitecture(p; globalpool = "deepset")
train(GNN, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN")

# WGNN estimator
seed!(1)
WGNN = gnnarchitecture(p; globalpool = "deepset", propagation = "WeightedGraphConv")
train(WGNN, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_WGNN")

# ---- Load the trained estimators ----

Flux.loadparams!(GNN,   loadbestweights(path * "/runs_GNN"))
Flux.loadparams!(WGNN,  loadbestweights(path * "/runs_WGNN"))


# ---- Assessment ----

# Construct the specific set of irregular locations, S
S = rand(n, 2)
D = pairwise(Euclidean(), S, S, dims = 1)
A = adjacencymatrix(D, ϵ = ϵ)
g = GNNGraph(A)
ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D

function assessestimators(θ, Z, ξ, g)

	Z = reshapedataGNN(Z, g)

	assessment = assess(
		[GNN, WGNN], θ, [Z];
		estimator_names = ["GNN", "WGNN"],
		parameter_names = ξ.parameter_names
	)

	return assessment
end

# Sample a large set of parameters for computing the risk function
seed!(1)
assessments = map(1:10) do i
	θ = Parameters(ξ, K_test)
	Z = simulate(θ, M)
	assessment = assessestimators(θ, Z, ξ, g)
	assessment.θandθ̂[:, :trial] .= i
	assessment.runtime[:, :trial] .= i
	assessment
end
assessment = merge(assessments...)
CSV.write(path * "/estimates_test.csv", assessment.θandθ̂)
CSV.write(path * "/runtime_test.csv", assessment.runtime)

# Focus on a small number of parameters for visualising the joint distribution
seed!(1)
θ = Parameters(ξ, 5)
Z = simulate(θ, M, 100)
assessment = assessestimators(θ, Z, ξ, g)
CSV.write(path * "/estimates_scenarios.csv", assessment.θandθ̂)
CSV.write(path * "/runtime_scenarios.csv", assessment.runtime)
