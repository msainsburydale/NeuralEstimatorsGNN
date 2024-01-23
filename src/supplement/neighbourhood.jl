# ------------------------------------------------------------------------------
#  Experiment: Definition of the node neighbourhood
#  - a disc of fixed spatial radius
#  - k-nearest neighbours for fixed k
#  - a random set of k neighbours selected within a disc of fixed spatial radius
# ------------------------------------------------------------------------------

using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
quick       = parsed_args["quick"]

model = joinpath("GP", "nuFixed")
m = 1
using NeuralEstimators
using NeuralEstimatorsGNN
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src", model, "model.jl"))
include(joinpath(pwd(), "src", "architecture.jl"))

path = "intermediates/supplement/neighbours"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 5
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
	K_val   = max(K_val, 100)
end
K_test = K_val

p = ξ.p
n = size(ξ.D, 1)

# For uniformly sampled locations on the unit square, the probability that a
# point falls within a circle of radius r << 1 centred away from the boundary is
# simply πr². So, on average, we expect nπr² neighbours for each node. Use this
# information to choose k in away that makes for a fair comparison.
n = 250  # sample size
r = 0.15 # disc radius
k = ceil(Int, n*π*r^2)

# Number of epochs used during training: Early stopping means that we never train
# for the full amount of epochs
epochs = quick ? 2 : 1000

# ---- initialise the estimators ----

seed!(1)
gnn1 = gnnarchitecture(p; propagation = "WeightedGraphConv")
gnn2 = deepcopy(gnn1)
gnn3 = deepcopy(gnn1)

# ---- Training ----

# Sample parameter vectors and simulate data
seed!(1)
θ_val   = Parameters(K_val,   ξ, n, J = 5)
θ_train = Parameters(K_train, ξ, n, J = 5)
Z_train = simulate(θ_train, m)
Z_val   = simulate(θ_val, m)

# Estimator trained with a fixed radius
θ̃_val   = modifyneighbourhood(θ_val, r)
θ̃_train = modifyneighbourhood(θ_train, r)
train(gnn1, θ̃_train, θ̃_val, Z_train, Z_val, savepath = joinpath(path, "fixedradius"), epochs = epochs)

# Estimator trained with k-nearest neighbours for fixed k
θ̃_val   = modifyneighbourhood(θ_val, k)
θ̃_train = modifyneighbourhood(θ_train, k)
train(gnn2, θ̃_train, θ̃_val, Z_train, Z_val, savepath = joinpath(path, "knearest"), epochs = epochs)

# Estimator trained with a random set of k neighbours selected within a disc of fixed spatial radius
θ̃_val   = modifyneighbourhood(θ_val, r, k)
θ̃_train = modifyneighbourhood(θ_train, r, k)
train(gnn3, θ̃_train, θ̃_val, Z_train, Z_val, savepath = joinpath(path, "combined"), epochs = epochs)

# ---- Load the trained estimators ----

Flux.loadparams!(gnn1,  loadbestweights(joinpath(path, "fixedradius")))
Flux.loadparams!(gnn2,  loadbestweights(joinpath(path, "knearest")))
Flux.loadparams!(gnn3,  loadbestweights(joinpath(path, "combined")))

# ---- Assess the estimators ----

function assessestimators(n, ξ, K::Integer)
	println("	Assessing estimators with n = $n...")

	# test parameter vectors for estimating the risk function
	seed!(1)
	θ = Parameters(K, ξ, n)

	# Estimator trained with a fixed radius
	θ̃ = modifyneighbourhood(θ, r)
	seed!(1); Z = simulate(θ̃, m)
	assessment = assess(
		[gnn1], θ̃, Z;
		estimator_names = ["fixedradius"],
		parameter_names = ξ.parameter_names
	)

	# Estimator trained with k-nearest neighbours for fixed k
	θ̃ = modifyneighbourhood(θ, k)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn2], θ̃, Z;
		estimator_names = ["knearest"],
		parameter_names = ξ.parameter_names
	))

	# Estimator trained with a random set of k neighbours selected within a disc of fixed spatial radius
	θ̃ = modifyneighbourhood(θ, r, k)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn3], θ̃, Z;
		estimator_names = ["combined"],
		parameter_names = ξ.parameter_names
	))

	# Add sample size information
	assessment.df[:, :n] .= n

	return assessment
end

assessestimators(n, ξ, K_test) # compile code to ensure accurate timings
test_n = [30, 60, 100, 200, 300, 500, 750, 1000]
assessment = [assessestimators(n, ξ, K_test) for n ∈ test_n]
assessment = merge(assessment...)
CSV.write(joinpath(path, "estimates.csv"), assessment.df)
CSV.write(joinpath(path, "runtime.csv"), assessment.runtime)


# ---- Accurately assess the run-time for a single data set ----

gnn1  = gnn1 |> gpu
gnn2  = gnn2 |> gpu
gnn3  = gnn3 |> gpu

function testruntime(n, ξ)

    # Simulate locations and data
	seed!(1)
    S = rand(n, 2)
	D = pairwise(Euclidean(), S, S, dims = 1)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new distance matrix D (needed for simulation and ML estimation)
	θ = Parameters(1, ξ)
	Z = simulate(θ, m; convert_to_graph = false)

	# Fixed radius
	g = adjacencymatrix(D, r)
	Z̃ = reshapedataGNN(Z, g)
	Z̃ = Z̃ |> gpu
	t_gnn1 = @belapsed gnn1($Z̃)

	# k-nearest neighbours
	g = adjacencymatrix(D, k)
	Z̃ = reshapedataGNN(Z, g)
	Z̃ = Z̃ |> gpu
	t_gnn2 = @belapsed gnn2($Z̃)

	# combined
	g = adjacencymatrix(D, r, k)
	Z̃ = reshapedataGNN(Z, g)
	Z̃ = Z̃ |> gpu
	t_gnn3 = @belapsed gnn3($Z̃)

	# Store the run times as a data frame
	DataFrame(time = [t_gnn1, t_gnn2, t_gnn3], estimator = ["fixedradius", "knearest", "combined"], n = n)
end

seed!(1)
times = []
for n ∈ test_n
	t = testruntime(n, ξ)
	push!(times, t)
end
times = vcat(times...)

CSV.write(joinpath(path, "runtime_singledataset.csv"), times)
