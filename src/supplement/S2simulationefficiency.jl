using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true

end
parsed_args = parse_args(arg_table)
quick       = parsed_args["quick"]

model=joinpath("GP", "nuSigmaFixed")
m=1
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using DataFrames
using GraphNeuralNetworks
using CSV
using CUDA

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/ML.jl"))
include(joinpath(pwd(), "src/architecture.jl"))
epochs = quick ? 2 : 200
p = ξ.p

path = "intermediates/supplement/simulationefficiency"
if !isdir(path) mkpath(path) end

# ---- Sample parameters ----

# In this experiment, we are assessing the simulation efficiency of the
# estimators. To do this, we will train and assess a series of estimators
# using an increasing number of data sets, up to a maximum of K.
K = 50000
if quick
	K = K ÷ 25
end
K_seq = 2000:2000:K

seed!(1)
n = 256
S = rand(n, 2) # fixed set of locations #TODO name this S_0
D = pairwise(Euclidean(), S, dims = 1)
ξ = merge(ξ, (S = S, D = D)) #TODO this is a case where we store S and D in xi

# ---- Estimators ----

seed!(1)
gnn1 = gnnarchitecture(p)
gnn2 = deepcopy(gnn1)
gnn3 = deepcopy(gnn1)

# ---- Training ----

seed!(1); θ₁_train = Parameters(K, ξ); θ₁_val = Parameters(K, ξ) #TODO this is a place where the "bootstrap" definition is needed
seed!(1); θ₂_train = Parameters(K, ξ, n, cluster_process = false); θ₂_val = Parameters(K, ξ, n, cluster_process = false)
seed!(1); θ₃_train = Parameters(K, ξ, n, cluster_process = true);  θ₃_val = Parameters(K, ξ, n, cluster_process = true)

## Code keeping Z fixed
function subsetandsimulate(θ_train, θ_val, indices, m)
  θ_t = subsetparameters(θ_train, indices)
  θ_v = subsetparameters(θ_val, indices)
  seed!(1)
  Z_t = simulate(θ_t, m)
  Z_v = simulate(θ_v, m)
  return θ_t, θ_v, Z_t, Z_v
end

for k ∈ K_seq
  @info "Training the estimators with K = $k..."
  if k != K_seq[1]
    loadpath = joinpath(path, "GNN1_K$(k-K_seq.step)", "best_network.bson")
    @load loadpath model_state
    Flux.loadmodel!(gnn1, model_state)

    loadpath = joinpath(path, "GNN2_K$(k-K_seq.step)", "best_network.bson")
    @load loadpath model_state
    Flux.loadmodel!(gnn2, model_state)

    loadpath = joinpath(path, "GNN3_K$(k-K_seq.step)", "best_network.bson")
    @load loadpath model_state
    Flux.loadmodel!(gnn3, model_state)
  end
  indices = 1:k
  train(gnn1, subsetandsimulate(θ₁_train, θ₁_val, indices, m)..., savepath = joinpath(path, "GNN1_K$k"), epochs = epochs)
  train(gnn2, subsetandsimulate(θ₂_train, θ₂_val, indices, m)..., savepath = joinpath(path, "GNN2_K$k"), epochs = epochs)
  train(gnn3, subsetandsimulate(θ₃_train, θ₃_val, indices, m)..., savepath = joinpath(path, "GNN3_K$k"), epochs = epochs)
end


# ---- Assessment with respect to the fixed locations S ----

seed!(1)
θ_test = Parameters(2500, ξ)
Z_test = simulate(θ_test, m)

assessments = []
for k ∈ K_seq
  @info "Assessing the estimators trained with K = $k..."
  
  loadpath = joinpath(path, "GNN1_K$k", "best_network.bson")
  @load loadpath model_state
  Flux.loadmodel!(gnn1, model_state)

  loadpath = joinpath(path, "GNN2_K$k", "best_network.bson")
  @load loadpath model_state
  Flux.loadmodel!(gnn2, model_state)

  loadpath = joinpath(path, "GNN3_K$k", "best_network.bson")
  @load loadpath model_state
  Flux.loadmodel!(gnn3, model_state)

  assessment = assess(
  		[gnn1, gnn2, gnn3], θ_test, Z_test;
  		estimator_names = ["Sfixed", "Srandom_uniform", "Srandom_cluster"],
  		parameter_names = ξ.parameter_names,
  		verbose = false
	 )
  assessment.df[:, :K] .= k
  push!(assessments, assessment)
end
assessment = merge(assessments...)

CSV.write(joinpath(path, "assessment.csv"), assessment.df)

@info "Finished!"
