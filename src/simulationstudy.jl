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
# skip_training = false
# quick=true

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using CSV
using DataFrames
using Distances
using GraphNeuralNetworks

include(joinpath(pwd(), "src/$model/model.jl"))
if model != "SPDE" include(joinpath(pwd(), "src/$model/ML.jl")) end
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

# ---- Estimators ----

seed!(1)
gnn = gnnarchitecture(p)

# ---- Training ----

J = 3

if !skip_training

	seed!(1)
	@info "simulating training data for the GNN..."
	θ_val,   Z_val   = variableirregularsetup(ξ, n, K = K_val, m = m, neighbour_parameter = r, J = J, clustering = true)
	θ_train, Z_train = variableirregularsetup(ξ, n, K = K_train, m = m, neighbour_parameter = r, J = J, clustering = true)
	@info "training the GNN..."
	trainx(gnn, θ_train, θ_val, Z_train, Z_val, savepath = path * "/runs_GNN", epochs = epochs, batchsize = 16)

end

# ---- Load the trained estimator ----

Flux.loadparams!(gnn,  loadbestweights(path * "/runs_GNN_m$M"))

# ---- Run-time assessment ----

if isdefined(Main, :ML)

	# Accurately assess the run-time for a single data set
	seed!(1)
	S = rand(n, 2)
	D = pairwise(Euclidean(), S, S, dims = 1)
	A = adjacencymatrix(D, r)
	g = GNNGraph(A)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new distance matrix D (needed for simulation and ML estimation)

	θ = Parameters(1, ξ, J = 1)
	Z = simulate(θ, M)

	θ₀ = mean.([ξ.Ω...])
	ξ = (ξ..., S = S, D = D, θ₀ = θ₀)
	tmap = @belapsed ML(Z, ξ)

	Z = reshapedataGNN(Z, g)
	Z = Z |> gpu
	gnn = gnn|> gpu
	tgnn = @belapsed gnn(Z)

	t = DataFrame((time = [tgnn, tmap], estimator = ["GNN", "ML"]))
	CSV.write(path * "/runtime.csv", t)

end

# ---- Assess the estimators ----


#TODO why is ML run twice?

function assessestimators(θ, Z, g, ξ)

	assessment = assess(
		[gnn], θ, reshapedataGNN(Z, g);
		estimator_names = ["GNN"],
		parameter_names = ξ.parameter_names
	)

	if isdefined(Main, :ML)
		assessment = merge(assessment, assess([ML], θ, Z; estimator_names = ["ML"], parameter_names = ξ.parameter_names, use_ξ = true, ξ = ξ))
	end

	return assessment
end

function assessestimators(S, ξ, K::Integer, set::String)

	D = pairwise(Euclidean(), S, S, dims = 1)
	A = adjacencymatrix(D, r)
	g = GNNGraph(A)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new distance matrix D (needed for simulation and ML estimation)

	# test set for estimating the risk function
	seed!(1)
	θ = Parameters(K_test, ξ)
	Z = simulate(θ, M)
	ξ = (ξ..., θ₀ = θ.θ)
	assessment = assessestimators(θ, Z, g, ξ)
	CSV.write(path * "/estimates_test_$set.csv", assessment.df)
	CSV.write(path * "/runtime_test_$set.csv", assessment.runtime)

	# small number of parameters for visualising the sampling distributions
	K_scenarios = 5
	seed!(1)
	θ = Parameters(K_scenarios, ξ)
	J = quick ? 10 : 100
	Z = simulate(θ, M, J)
	ξ = (ξ..., θ₀ = θ.θ)
	assessment = assessestimators(θ, Z, g, ξ)
	CSV.write(path * "/estimates_scenarios_$set.csv", assessment.df)
	CSV.write(path * "/runtime_scenarios_$set.csv", assessment.runtime)

	# save spatial fields for plotting
	Z = Z[1:K_scenarios] # only need one field per parameter configuration
	colons  = ntuple(_ -> (:), ndims(Z[1]) - 1)
	z  = broadcast(z -> vec(z[colons..., 1]), Z) # save only the first replicate of each parameter configuration
	z  = vcat(z...)
	z  = broadcast(ξ.invtransform, z)
	d  = prod(size(Z[1])[1:end-1])
	k  = repeat(1:K_scenarios, inner = d)
	s1 = repeat(S[:, 1], K_scenarios)
	s2 = repeat(S[:, 2], K_scenarios)
	df = DataFrame(Z = z, k = k, s1 = s1, s2 = s2)
	CSV.write(path * "/Z_$set.csv", df)

	return 0
end

# Test with respect to a set of irregular uniformly sampled locations
seed!(1)
set = "uniform"
S = rand(n, 2)
seed!(1)
assessestimators(S, ξ, K_test, set)

# Test with respect to locations sampled only in the first and third quadrants
#         . . .
#         . . .
#         . . .
#  . . .
#  . . .
#  . . .
seed!(1)
set = "quadrants"
S₁ = 0.5 * rand(n÷2, 2)
S₂ = 0.5 * rand(n÷2, 2) .+ 0.5
S  = vcat(S₁, S₂)
seed!(1)
assessestimators(S, ξ, K_test, set)


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
set = "mixedsparsity"
n_centre = (3 * n) ÷ 4
n_corner = (n - n_centre) ÷ 4
S_centre  = 1/3 * rand(n_centre, 2) .+ 1/3
S_corner1 = 1/3 * rand(n_corner, 2)
S_corner2 = 1/3 * rand(n_corner, 2); S_corner2[:, 2] .+= 2/3
S_corner3 = 1/3 * rand(n_corner, 2); S_corner3[:, 1] .+= 2/3
S_corner4 = 1/3 * rand(n_corner, 2); S_corner4 .+= 2/3
S = vcat(S_centre, S_corner1, S_corner2, S_corner3, S_corner4)
seed!(1)
assessestimators(S, ξ, K_test, set)


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
set = "cup"
n_strip2 = n÷3 + n % 3 # ensure that total sample size is n (even if n is not divisible by 3)
S_strip1 = rand(n÷3, 2);      S_strip1[:, 1] .*= 0.2;
S_strip2 = rand(n_strip2, 2); S_strip2[:, 1] .*= 0.6; S_strip2[:, 1] .+= 0.2; S_strip2[:, 2] .*= 1/3;
S_strip3 = rand(n÷3, 2);      S_strip3[:, 1] .*= 0.2; S_strip3[:, 1] .+= 0.8;
S = vcat(S_strip1, S_strip2, S_strip3)
seed!(1)
assessestimators(S, ξ, K_test, set)