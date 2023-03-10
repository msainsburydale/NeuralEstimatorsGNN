using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
quick       = parsed_args["quick"]

using NeuralEstimators
using NeuralEstimatorsGNN
using Distances: pairwise, Euclidean
using GraphNeuralNetworks
using CSV

model  = "GaussianProcess/fourparameters"
m      = 1 # number of replicates per spatial field
n      = 512 # number of observations per field during training
neighbours = 8 # number of neighbours to consider

include(joinpath(pwd(), "src/$model/Parameters.jl"))
include(joinpath(pwd(), "src/$model/Simulation.jl"))
include(joinpath(pwd(), "src/Architecture.jl"))

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 20000
K_val   = K_train ÷ 10
K_test  = K_val
if quick
	K_train = K_train ÷ 100
	K_val   = K_val ÷ 100
	K_test  = K_test ÷ 100
end


function variableirregularsetup(ξ; K, n, J = 5, m = 1)

	D = map(1:K) do k
		S = rand(n, 2)
		D = pairwise(Euclidean(), S, S, dims = 1)
		D
	end
	A = adjacencymatrix.(D, neighbours)
	g = GNNGraph.(A)

	ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D
	θ = Parameters(ξ, K, J = J)
	Z = simulate(θ, m)

	g = repeat(g, inner = J)
	Z = reshapedataGNN(Z, g)

	return θ, Z
end


# ---- Training ----

# Construct a variable set of irregular locations {Sₖ : k = 1, …, K}
#TODO could do on-the-fly simulation to improve results
θ_val,   Z_val    = variableirregularsetup(ξ, K = K_val, n = n)
θ_train, Z_train  = variableirregularsetup(ξ, K = K_train, n = n)

# Train the estimator
seed!(1)
GNN = gnnarchitecture(ξ.p; globalpool = "deepset", propagation = "WeightedGraphConv")
train(GNN, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs")

# Load the trained estimator
Flux.loadparams!(GNN,  loadbestweights(path * "/run"))


# ---- Assessment: training n ----

#TODO use a variable set of locations (possibly using variableirregularsetup)

# Construct a set of irregular locations, S
S = rand(n, 2)
D = pairwise(Euclidean(), S, S, dims = 1)
A = adjacencymatrix(D, neighbours)
g = GNNGraph(A)
ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D

# Sample a large set of parameters for computing the risk function
seed!(1)
θ = Parameters(ξ, K_test)
Z = simulate(θ, m)
Z = reshapedataGNN(Z, g)
assessment = assess(
	[GNN], θ, [Z];
	estimator_names = ["GNN"],
	parameter_names = ξ.parameter_names
)
CSV.write(path * "/estimates_test.csv", assessment.θandθ̂)
CSV.write(path * "/runtime_test.csv", assessment.runtime)

# Focus on a small number of parameters for visualising the joint distribution
seed!(1)
θ = Parameters(ξ, 5)
Z = simulate(θ, m, 100)
Z = reshapedataGNN(Z, g)
assessment = assess(
	[GNN], θ, [Z];
	estimator_names = ["GNN"],
	parameter_names = ξ.parameter_names
)
CSV.write(path * "/estimates_scenarios.csv", assessment.θandθ̂)
CSV.write(path * "/runtime_scenarios.csv", assessment.runtime)


# ---- Assessment: larger n ----


# ---- Apply the estimator to the massive spatial data set ----
