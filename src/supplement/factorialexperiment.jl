# -------------------------------------------------------------------
# ---- Experiment: GNNs in the presence of variable sample sizes ----
# -------------------------------------------------------------------

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

path = "intermediates/supplement/factorialexperiment/$model"
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

# Sample parameters used during training
seed!(1)
J = 3
n = 1000
θ_val   = Parameters(K_val,   ξ, n, J = J)
θ_train = Parameters(K_train, ξ, n, J = J)

for nlayers ∈ [1, 2, 3, 4, 5] # number of propagation layers (in addition to the first layer)
	@info "Training GNN with $(nlayers) propagation layers"
	for nh ∈ [4, 8, 16, 32, 64, 128, 256] # number of channels in each propagation layer
		@info "Training GNN with $(nh) channels in each propagation layer"
		seed!(1)
		gnn = gnnarchitecture(p, nlayers = nlayers, nh = nh)
		train(gnn, θ_train, θ_val, simulate, m = M, savepath = path * "/runs_GNN_depth$(nlayers)_width$(nh)", epochs = epochs, epochs_per_Z_refresh = 3)
	end
end
