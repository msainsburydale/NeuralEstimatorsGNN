# ------------------------------------------------------------------------------
#  Comparing definitions for the neighbourhood of a node:
#  - Disc of fixed spatial radius r
#  - a subset of k neighbours within a disc of fixed spatial radius r
#  - k-nearest neighbours
#  - k-nearest neighbours subject to a maxmin ordering
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

model = joinpath("GP", "nuSigmaTauFixed")
m = 1
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using DataFrames
using GraphNeuralNetworks
using CSV
using CUDA

path = joinpath("intermediates", "supplement", "neighbours")
if !isdir(path) mkpath(path) end

# Helper functions for this script, with the same API as adjacencymatrix()
function modifyneighbourhood(θ::Parameters, args...; kwargs...)
  S = θ.locations
	if !(typeof(S) <: AbstractVector) S = [S] end
	S = broadcast.(Float32, S)
  A = adjacencymatrix.(S, args...; kwargs...)
	graphs = GNNGraph.(A)
	Parameters(θ.θ, S, graphs, θ.chols, θ.chol_pointer, θ.loc_pointer)
end

include(joinpath(pwd(), "src", model, "model.jl"))
include(joinpath(pwd(), "src", "architecture.jl"))
p = ξ.p # number of parameters in the statistical model
cluster_process = true

function gnnarchitecture(
	p::Integer;
	nₕ = [128, 128, 128, 128],    # number of channels in each propagation layer
	aggr = mean,                  # neighbourhood aggregation function
	final_activation = identity
	)

	if isa(nₕ, Integer) nₕ = [nₕ] end
	nlayers = length(nₕ)              # number of propagation layers
	w_channels = [16, fill(1, nlayers-1)...]   # number of channels for weight function w(⋅)
	in=[1, nₕ[1:end-1]...]            # input dimensions of propagation features

	# Propagation module
	propagation_layers = map(1:nlayers) do l
		SpatialGraphConv(in[l] => nₕ[l], relu, w_scalar = true, w_exponential_decay = true, aggr = aggr, w_channels = w_channels[l])
	end
	propagation = GNNChain(propagation_layers...)

	# Readout module
	readout = GlobalPool(mean)
	nᵣ = nₕ[end] # dimension of readout vector

	# Summary network
	ψ = GNNSummary(propagation, readout)

	# Mapping module
	ϕ = Chain(
	  Dense(nᵣ => 128, relu), 
	  Dense(nᵣ => 128, relu), 
	  Dense(128 => p, final_activation)
	  )

	return DeepSet(ψ, ϕ)
end


# ---- Sample training parameter vectors ----

# Size of the training set
K = quick ? 1000 : 15000
J = 5
n = 150:350  # sample sizes

seed!(1)
@info "Sampling parameter vectors used for validation..."
θ_val   = Parameters(K ÷ 10, ξ, n, J = J, cluster_process = cluster_process)
@info "Sampling parameter vectors used for training..."
θ_train = Parameters(K, ξ, n, J = J, cluster_process = cluster_process)

# ---- Training ----

epochs_per_Z_refresh = 3
epochs = quick ? 2 : 50
stopping_epochs = epochs   # "turn off" early stopping entirely by setting equal to epochs

all_k = [3, 5, 10, 15, 20, 25, 30] 
all_r = [0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.15]

for r ∈ all_r
  	@info "Training GNN with disc of fixed radius r=$r"
  	savepath = joinpath(path, "fixedradius_r$r")
  	global θ_val   = modifyneighbourhood(θ_val, r)
  	global θ_train = modifyneighbourhood(θ_train, r)
  	seed!(1)
  	gnn = gnnarchitecture(p)
  	train(gnn, θ_train, θ_val, simulate, m = m, savepath = savepath,
  		  epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh,
  		  stopping_epochs = stopping_epochs)
end

for k ∈ all_k
  for maxmin ∈ [true, false]
  	@info "Training GNN with $(maxmin ? "k-nearest neighbours subject to a maxmin ordering" : "k-nearest neighbours") and k=$k"
  	savepath = joinpath(path, "$(maxmin ? "maxmin" : "knearest")_k$k")
  	global θ_val   = modifyneighbourhood(θ_val, k; maxmin = maxmin)
  	global θ_train = modifyneighbourhood(θ_train, k; maxmin = maxmin)
  	seed!(1)
  	gnn = gnnarchitecture(p)
  	train(gnn, θ_train, θ_val, simulate, m = m, savepath = savepath,
  		  epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh,
  		  stopping_epochs = stopping_epochs)
	end
end

for r ∈ [0.15]
    for k ∈ all_k
    	@info "Training GNN with disc of fixed radius r=$r and maximum number of neighbours k=$k"
    	savepath = joinpath(path, "fixedradiusmaxk_r$(r)_k$(k)")
    	global θ_val   = modifyneighbourhood(θ_val, r, k; random = false)
    	global θ_train = modifyneighbourhood(θ_train, r, k; random = false)
    	seed!(1)
    	gnn = gnnarchitecture(p)
    	train(gnn, θ_train, θ_val, simulate, m = m, savepath = savepath,
    		  epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh,
    		  stopping_epochs = stopping_epochs)
  	end
end

# ---- Assessment ----

# Number of data sets and sample sizes to use during testing
K_test = quick ? 50 : 1000
n_test = [60, 100, 200, 250, 300, 350, 400, 500, 600, 750, 1000]

assessments = []
for n ∈ n_test
	@info "Assessing the estimators with sample size n=$n"
	seed!(1)
	θ_test   = Parameters(K_test, ξ, n, J = 1) 
	θ_single = Parameters(1, ξ, n, J = 1)
  for k ∈ all_k
  	@info "Assessing the estimator with k=$k"
  	for maxmin ∈ [true, false]
  	  gnn = gnnarchitecture(p)
  		loadpath = joinpath(path, "$(maxmin ? "maxmin" : "knearest")_k$k", "best_network.bson")
		@load loadpath model_state
		Flux.loadmodel!(gnn, model_state)
    
  		θ_test = modifyneighbourhood(θ_test, k; maxmin = maxmin)
  		θ_single = modifyneighbourhood(θ_single, k; maxmin = maxmin)
    
  		# Assess the estimator's accuracy
  		seed!(1)
  		Z_test = simulate(θ_test, m)
  		assessment = assess(
  			[gnn], θ_test, Z_test;
  			estimator_name = "$(maxmin ? "maxmin" : "knearest")",
  			parameter_names = ξ.parameter_names,
  			verbose = false
  		)
  		allowmissing!(assessment.df)
  		assessment.df[:, :k] .= k
  		assessment.df[:, :r] .= missing
  		assessment.df[:, :n] .= n
    
  		# Accurately assess the inference time for a single data set
  	  gnn = gnn |> gpu
  	  Z_single = simulate(θ_single, m) |> gpu
  	  t = @belapsed $gnn($Z_single)
  	  assessment.df[:, :inference_time] .= t
    
  	  push!(assessments, assessment)
  	end
  end
	for r ∈ all_r
		@info "Assessing the estimator with r=$r"
		gnn = gnnarchitecture(p)
		loadpath = joinpath(path, "fixedradius_r$r", "best_network.bson")
		@load loadpath model_state
		Flux.loadmodel!(gnn, model_state)
  
		θ_test = modifyneighbourhood(θ_test, r)
		θ_single = modifyneighbourhood(θ_single, r)
  
		# Assess the estimator's accuracy
		seed!(1)
		Z_test = simulate(θ_test, m)
		assessment = assess(
			[gnn], θ_test, Z_test;
			estimator_name = "fixedradius",
			parameter_names = ξ.parameter_names,
			verbose = false
		)
		allowmissing!(assessment.df)
		assessment.df[:, :k] .= missing 
		assessment.df[:, :r] .= r 
		assessment.df[:, :n] .= n
  
		# Accurately assess the inference time for a single data set
	 gnn = gnn |> gpu
	  Z_single = simulate(θ_single, m) |> gpu
	 t = @belapsed $gnn($Z_single)
	 assessment.df[:, :inference_time] .= t
	 push!(assessments, assessment)
	end
	for r ∈ [0.15] # TODO error prone, define a variable for this
		for k ∈ all_k
			@info "Assessing the estimator with r=$r and k=$k"
			gnn = gnnarchitecture(p)
			loadpath = joinpath(path, "fixedradiusmaxk_r$(r)_k$(k)", "best_network.bson")
			@load loadpath model_state
			Flux.loadmodel!(gnn, model_state)
		
			θ_test = modifyneighbourhood(θ_test, r, k)
			θ_single = modifyneighbourhood(θ_single, r, k)
		
			# Assess the estimator's accuracy
			seed!(1)
			Z_test = simulate(θ_test, m)
			assessment = assess(
				[gnn], θ_test, Z_test;
				estimator_name = "fixedradiusmaxk",
				parameter_names = ξ.parameter_names,
				verbose = false
			)
			allowmissing!(assessment.df)
			assessment.df[:, :k] .= k
			assessment.df[:, :r] .= r 
			assessment.df[:, :n] .= n
		
			# Accurately assess the inference time for a single data set
		gnn = gnn |> gpu
			Z_single = simulate(θ_single, m) |> gpu
		t = @belapsed $gnn($Z_single)
		assessment.df[:, :inference_time] .= t
		push!(assessments, assessment)
		end
	end
end
assessment = merge(assessments...)
CSV.write(joinpath(path, "sensitivity_analysis.csv"), assessment.df) 
#