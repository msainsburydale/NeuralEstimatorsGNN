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
using NamedArrays
using BenchmarkTools
using CSV
using DataFrames
using Distances
using GraphNeuralNetworks

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/architecture.jl"))
include(joinpath(pwd(), "src/$model/ML.jl"))
p = ξ.p
n = ξ.n

path = "intermediates/$model"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 5
if quick
	K_train = K_train ÷ 10
	K_val   = K_val   ÷ 10
end


epochs = quick ? 20 : 200
J = 5

if !skip_training
	@info "Generating training data..."
	seed!(1)
	@info "Sampling parameter vectors used for validation..."
	θ_val = Parameters(K_val, ξ, n, J = J)
	@info "Sampling parameter vectors used for training..."
	θ_train = Parameters(K_train, ξ, n, J = J)
	@info "Training the GNN point estimator..."
end

# -----------------------------------------------------------------------------
# -------------------------- Point estimator ----------------------------------
# -----------------------------------------------------------------------------

@info "Constructing and assessing neural point estimator..."

seed!(1)
pointestimator = gnnarchitecture(p)

if !skip_training
	trainx(pointestimator, θ_train, θ_val, simulate, m, savepath = path * "/runs_GNN", epochs = epochs, batchsize = 16, epochs_per_Z_refresh = 3)
end

# Load the trained estimator
Flux.loadparams!(pointestimator,  loadbestweights(joinpath(path, "runs_GNN_m$M")))

# ---- Run-time assessment ----

# Accurately assess the run-time for a single data set
if isdefined(Main, :ML)

	# Simulate data
	seed!(1)
	S = rand(n, 2)
	D = pairwise(Euclidean(), S, dims = 1)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new spatial locations and distance matrix (the latter is needed for ML estimation)
	θ = Parameters(1, ξ)
	Z = simulate(θ, M; convert_to_graph = false)

	# ML estimates (initialised to the prior mean)
	θ₀ = mean.([ξ.Ω...])
	ξ = (ξ..., θ₀ = θ₀)
	t_ml = @belapsed ML(Z, ξ)

	# GNN estimates
	g = θ.graphs[1]
	Z = reshapedataGNN(Z, g)
	Z = Z |> gpu
	pointestimator  = pointestimator|> gpu
	t_gnn = @belapsed pointestimator(Z)

	# Save the runtime
	t = DataFrame(time = [t_gnn, t_ml], estimator = ["GNN", "ML"])
	CSV.write(path * "/runtime.csv", t)

end

# ---- Assess the point estimators ----

K_test = quick ? 100 : 1000

function assessestimators(θ, Z, ξ)

	# Convert the data to a graph
	g = θ.graphs[1]
	Z_graph = reshapedataGNN(Z, g)

	# Assess the GNN
	assessment = assess(pointestimator, θ, Z_graph; estimator_name = "GNN", parameter_names = ξ.parameter_names)

	# Assess the ML estimator (if it is defined)
	if isdefined(Main, :ML)
		ξ = (ξ..., θ₀ = θ.θ)
		assessment = merge(assessment, assess(ML, θ, Z; estimator_name = "ML", ξ = ξ))
	end

	return assessment
end

function assessestimators(ξ, set::String)

	# Generate spatial locations and construct distance matrix
	S = spatialconfigurations(n, set)
	D = pairwise(Euclidean(), S, S, dims = 1)
	ξ = (ξ..., S = S, D = D) # update ξ to contain the new spatial locations and distance matrix (the latter is needed for ML estimation)

	# test set for estimating the risk function
	θ = Parameters(K_test, ξ)
	Z = simulate(θ, M, convert_to_graph = false)
	assessment = assessestimators(θ, Z, ξ)
	CSV.write(path * "/estimates_test_$set.csv", assessment.df)
	CSV.write(path * "/runtime_test_$set.csv", assessment.runtime)

	# small number of parameters for visualising the sampling distributions
	K_scenarios = 5
	seed!(1) # Important that these Parameter scenarios are the same for all locations when constructing the plots.
	θ = Parameters(K_scenarios, ξ)
	J = quick ? 10 : 100
	Z = simulate(θ, M, J, convert_to_graph = false)
	assessment = assessestimators(θ, Z, ξ)
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

	return assessment
end


# Test with respect to a set of uniformly sampled locations
#  .   . . . .
#  . . . . .
#  . . . . . .
#  .   . .   .
#  . . . . . .
#  .   . . .
seed!(1)
assessestimators(ξ, "uniform")

# Test with respect to locations sampled only in the first and third quadrants
#         . . .
#         . . .
#         . . .
#  . . .
#  . . .
#  . . .
seed!(1)
assessestimators(ξ, "quadrants")


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
assessestimators(ξ, "mixedsparsity")


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
assessestimators(ξ, "cup")



# -----------------------------------------------------------------------------
# --------------------- Uncertainty quantification ----------------------------
# -----------------------------------------------------------------------------

@info "Constructing and assessing neural quantile estimator..."

# point estimator:
pointestimator = gnnarchitecture(p)
pointestimator_path = joinpath(path, "runs_GNN_m$M")
Flux.loadparams!(pointestimator, loadbestweights(pointestimator_path))

# Credible-interval estimator:
seed!(1)
v = gnnarchitecture(p; final_activation = identity)
Flux.loadparams!(v, loadbestweights(pointestimator_path)) # pretrain with point estimator
intervalestimator = IntervalEstimator(v)

if !skip_training
	@info "training the GNN quantile estimator for marginal posterior credible intervals..."
	trainx(intervalestimator, θ_train, θ_val, simulate, m, savepath = joinpath(path, "runs_GNN_CI"), epochs = epochs, batchsize = 16, epochs_per_Z_refresh = 3)
end

Flux.loadparams!(intervalestimator, loadbestweights(joinpath(path, "runs_GNN_CI_m$M")))


# ---- Empirical coverage ----

# Simulate test data
seed!(2023)
K_test = quick ? 100 : 3000
θ_test = Parameters(K_test, ξ, n, J = 3)
Z_test = simulate(θ_test, M)

# Assessment: Quantile estimator
assessment = assess(intervalestimator, θ_test, Z_test, estimator_name = "quantile", parameter_names = ξ.parameter_names)

# Assessment: Parametric Bootstrap
B = quick ? 50 : 500
θ̂_test = estimateinbatches(pointestimator, Z_test)
θ̂_test = Float64.(θ̂_test)
θ̂_test = max.(θ̂_test, 0.01)
θ̂_test = Parameters(θ̂_test, θ_test.locations, ξ)
Z_boot = [simulate(θ̂_test, M) for _ ∈ 1:B]
Z_boot = map(1:K_test) do k
	[z[k] for z ∈ Z_boot]
end
assessment_boot = assess(pointestimator, θ_test, Z_test, boot = Z_boot, estimator_name = "bootstrap_parametric", parameter_names = ξ.parameter_names)

# Assessment: Nonparametric Bootstrap
# if M > 1
#  # TODO subsetting the graph with subsetdata() is super slow... would be good to fix this so that non-parametric bootstrap is more efficient/can be used here.
#  assessment_boot2 = assess(pointestimator, θ_test, Z_test, boot = true, estimator_name = "bootstrap_nonparametric", parameter_names = ξ.parameter_names)
#  assessment_boot = merge(assessment_boot, assessment_boot2)
#end

# Save interval estimates
ass = merge(assessment, assessment_boot)
CSV.write(joinpath(path, "uq_interval_estimates.csv"), ass.df)

# Compute and save diagnostics
cov = vcat(coverage(assessment), coverage(assessment_boot))
is  = vcat(intervalscore(assessment), intervalscore(assessment_boot))
uq_assessment = innerjoin(cov, is, on = [:estimator, :parameter])
CSV.write(joinpath(path, "uq_assessment.csv"), uq_assessment)



# ---- Run-time assessment ----

# Accurately assess the run-time for a single data set
# Use a uniform process so that we can specify n exactly, rather than simply E(n)
θ = Parameters(1, ξ, n; cluster_process = false)
S = θ.locations
Z = simulate(θ, M)
Z = Z |> gpu

# Quantile estimator
intervalestimator = intervalestimator |> gpu
t1 = @belapsed intervalestimator(Z)

# Bootstrap
function bs(pointestimator, Z, S, B, ξ)
	θ̂ = pointestimator(Z)
	θ̂ = θ̂ |> cpu
	θ̂ = Parameters(θ̂, S, ξ)
	Z_boot = [simulate(θ̂, M)[1] for _ ∈ 1:B]
	estimateinbatches(pointestimator, Z_boot)
end
pointestimator = pointestimator |> gpu
t2 = @belapsed bs($pointestimator, $Z, $S, $B, $ξ)

# Save the runtime
t = DataFrame(time = [t1, t2], estimator = ["quantile", "bootstrap_parametric"])
CSV.write(joinpath(path, "uq_runtime.csv"), t)
