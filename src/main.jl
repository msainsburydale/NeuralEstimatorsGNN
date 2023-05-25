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

# model="Schlather"
# m=[1, 30]
# skip_training = false
# quick=true

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/MAP.jl")) # TODO add Pairwise likelihood for Schlather model (prototype with office PC)
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 5
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
end
K_test = K_val

p = ξ.p
n = size(ξ.D, 1)
r = ξ.r

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 1000

# ---- Estimators ----

seed!(1)
cnn = DeepSet(cnnarchitecture(p)...)
gnn = gnnarchitecture(p)

# ---- Training ----

# Note that we have to use different training data for the two estimators,
# because CNNs require gridded data whilst GNNs require training with irregular
# data if they are to generalise well.

if !skip_training

	# CNN estimator
	@info "training the CNN..."
	seed!(1)
	θ_val   = Parameters(K_val, ξ, J = 5)
	θ_train = Parameters(K_train, ξ, J = 5)
	Z_val   = simulate(θ_val, M)
	Z_train = simulate(θ_train, M)
	trainx(cnn, θ_train, θ_val, reshapedataCNN(Z_train), reshapedataCNN(Z_val), savepath = path * "/runs_CNN", epochs = epochs)

	# GNN estimator
	@info "training the GNN..."
	seed!(1)
	θ_val,   Z_val   = variableirregularsetup(ξ, n, K = K_val, m = m, neighbour_parameter = r)
	θ_train, Z_train = variableirregularsetup(ξ, n, K = K_train, m = m, neighbour_parameter = r)
	trainx(gnn, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN", epochs = epochs)
end



# ---- Load the trained estimators ----

Flux.loadparams!(cnn,  loadbestweights(path * "/runs_CNN_m$M"))
Flux.loadparams!(gnn,  loadbestweights(path * "/runs_GNN_m$M"))


# ---- Assess the estimators ----

function assessestimators(θ, Z, g, ξ; assess_CNN::Bool = false, assess_MAP::Bool = true)
	assessment = assess(
		[gnn], θ, reshapedataGNN(Z, g);
		estimator_names = ["GNN"],
		parameter_names = ξ.parameter_names
	)

	if assess_CNN
		assessment = merge(assessment, assess([cnn], θ, reshapedataCNN(Z); estimator_names = ["CNN"], parameter_names = ξ.parameter_names))
	end

	if assess_MAP
		assessment = merge(assessment, assess([MAP], θ, Z; estimator_names = ["MAP"], parameter_names = ξ.parameter_names, use_gpu = false, use_ξ = true, ξ = ξ))
	end

	return assessment
end

function assessestimators(S, ξ, K::Integer, set::String)

	D = pairwise(Euclidean(), S, S, dims = 1)
	A = adjacencymatrix(D, r)
	g = GNNGraph(A)
	ξ = (ξ..., D = D) # update ξ to contain the new distance matrix D (needed for simulation and MAP estimation)

	assess_CNN = set == "gridded"
	assess_MAP = isdefined(Main, :MAP)

	# test set for estimating the risk function
	seed!(1)
	θ = Parameters(K_test, ξ)
	Z = simulate(θ, M)
	ξ = (ξ..., θ₀ = θ.θ)
	assessment = assessestimators(θ, Z, g, ξ; assess_CNN = assess_CNN)
	CSV.write(path * "/estimates_test_$set.csv", assessment.df)
	CSV.write(path * "/runtime_test_$set.csv", assessment.runtime)

	# small number of parameters for visualising the sampling distributions
	K_scenarios = 5
	seed!(1)
	θ = Parameters(K_scenarios, ξ)
	Z = simulate(θ, M, 100)
	ξ = (ξ..., θ₀ = θ.θ)
	assessment = assessestimators(θ, Z, g, ξ; assess_CNN = assess_CNN)
	CSV.write(path * "/estimates_scenarios_$set.csv", assessment.df)
	CSV.write(path * "/runtime_scenarios_$set.csv", assessment.runtime)

	# save spatial fields for plotting
	Z = Z[1:K_scenarios] # only need one field per parameter configuration
	colons  = ntuple(_ -> (:), ndims(Z[1]) - 1)
	z  = broadcast(z -> vec(z[colons..., 1]), Z) # save only the first replicate of each parameter configuration
	z  = vcat(z...)
	d  = prod(size(Z[1])[1:end-1])
	k  = repeat(1:K_scenarios, inner = d)
	s1 = repeat(S[:, 1], K_scenarios)
	s2 = repeat(S[:, 2], K_scenarios)
	df = DataFrame(Z = z, k = k, s1 = s1, s2 = s2)
	CSV.write(path * "/Z_$set.csv", df)

	return 0
end


# Test with respect to gridded data
pts = range(0, 1, length = 16)
S   = expandgrid(pts, pts)
seed!(1)
assessestimators(S, ξ, K_test, "gridded")

# Test with respect to a set of irregular uniformly sampled locations
seed!(1)
S = rand(n, 2)
assessestimators(S, ξ, K_test, "uniform")

# Test with respect to locations sampled only in the first and third quadrants
#         . . .
#         . . .
#         . . .
#  . . .
#  . . .
#  . . .
seed!(1)
S₁ = 0.5 * rand(n÷2, 2)
S₂ = 0.5 * rand(n÷2, 2) .+ 0.5
S  = vcat(S₁, S₂)
assessestimators(S, ξ, K_test, "quadrants")


# Test with respect to locations with mixed sparsity.
# Divide the domain into 9 cells: have the
# central cell being very densely populated; the corner cells sparse; and the
# side cells empty.
# . .         . .
#   . .       .   .
#               .
#       . . .
#       . . .
#       . . .
# . .         .
#    .        . . .
# .             .
seed!(1)
n_centre = 200
@assert n_centre < n
@assert (n - n_centre) % 4 == 0
n_corner = (n - n_centre) ÷ 4
S_centre  = 1/3 * rand(n_centre, 2) .+ 1/3
S_corner1 = 1/3 * rand(n_corner, 2)
S_corner2 = 1/3 * rand(n_corner, 2); S_corner2[:, 2] .+= 2/3
S_corner3 = 1/3 * rand(n_corner, 2); S_corner3[:, 1] .+= 2/3
S_corner4 = 1/3 * rand(n_corner, 2); S_corner4 .+= 2/3
S = vcat(S_centre, S_corner1, S_corner2, S_corner3, S_corner4)
assessestimators(S, ξ, K_test, "mixedsparsity")


# Test with respect to locations with a cup shape ∪.
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . .           . .
# . . . . . . . . .
# . . . . . . . . .
# . . . . . . . . .
#Construct by considering the domain split into three vertical strips
seed!(1)
n_strip2 = n÷3 + n % 3 # ensure that total sample size is n (even if n is not divisible by 3)
S_strip1 = rand(n÷3, 2);      S_strip1[:, 1] .*= 0.2;
S_strip2 = rand(n_strip2, 2); S_strip2[:, 1] .*= 0.6; S_strip2[:, 1] .+= 0.2; S_strip2[:, 2] .*= 1/3;
S_strip3 = rand(n÷3, 2);      S_strip3[:, 1] .*= 0.2; S_strip3[:, 1] .+= 0.8;
S = vcat(S_strip1, S_strip2, S_strip3)
assessestimators(S, ξ, K_test, "cup")