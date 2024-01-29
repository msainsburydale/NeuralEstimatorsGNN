# -----------------------------------------------------------------
# ---- Experiment: neural-network architecture hyperparameters ----
# -----------------------------------------------------------------

using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true

end
parsed_args = parse_args(arg_table)
quick           = parsed_args["quick"]

model="GP/nuFixed"
m=[1]

M = maximum(m)
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src/$model/model.jl"))
include(joinpath(pwd(), "src/$model/ML.jl"))
include(joinpath(pwd(), "src/architecture.jl"))

path = "intermediates/supplement/discradius"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 10
if quick
	K_train = K_train ÷ 10
	K_val   = K_val   ÷ 10
	K_val   = max(K_val, 10)
end
K_test = K_val

p = ξ.p

# The number of epochs used during training: note that early stopping means that
# we never really train for the full amount of epochs
epochs = quick ? 2 : 200

J = 3
n = 250

for radius ∈ [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # number of channels in each propagation layer
  
  @info "Training GNN with disc radius = $(radius)"
  seed!(1)
  global ξ = merge(ξ, (δ = radius, ))
  θ_val   = Parameters(K_val,   ξ, n, J = J)
  θ_train = Parameters(K_train, ξ, n, J = J)
  θ_single = Parameters(1, ξ, n, J = 1)
  Z_single = simulate(θ_single, M) |> gpu
  
  for nlayers ∈ [1, 2, 3, 4, 5] # number of propagation layers (in addition to the first layer) #TODO why do we need the parenthetic comment?
	  @info "Training GNN with $(nlayers) propagation layers"
		
		seed!(1)
		gnn = gnnarchitecture(p, nlayers = nlayers)
		savepath = joinpath(path, "runs_GNN_depth$(nlayers)_radius$(radius)")
		train(gnn, θ_train, θ_val, simulate, m = M, savepath = savepath, epochs = epochs, epochs_per_Z_refresh = 3)
		
		# Accurately assess the inference time for a single data set
		Flux.loadparams!(gnn,  loadbestweights(savepath))
	  gnn  = gnn|> gpu
	  t = @belapsed $gnn($Z_single)
	  t = DataFrame(time = [t])
	  CSV.write(joinpath(savepath, "inference_time.csv"), t)
	end
end
