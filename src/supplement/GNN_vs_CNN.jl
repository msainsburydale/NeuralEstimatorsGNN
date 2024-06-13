using NeuralEstimators, Flux, GraphNeuralNetworks, Distances, Distributions, Folds, LinearAlgebra, Optim, Statistics, CairoMakie, CSV
using BSON: @save, @load
using Flux: flatten

fixedlocs = false   # use fixed locations (a 16x16 grid) or variable locations from a spatial point process? 
covfun = "matern"  # "matern" or "exponential" covariance function?
noise = true       # estimate noise parameter?

struct Parameters{T} <: ParameterConfigurations
	θ::Matrix{T}
	chols
	S
end

# Construct prior 
Π = (ρ = Uniform(0.05, 0.5), )
if noise 
	Π = (Π..., τ = Uniform(0, 1))
end
parameter_names = String.(collect(keys(Π)))

function covariancematrix(D; ρ, τ = nothing, covfun)
	if covfun == "exponential"
		Σ = exp.(-D ./ ρ)
	elseif covfun == "matern"
		ν = one(typeof(ρ))
		Σ = matern.(D, ρ, ν)
	else
		error("unrecognised covfun")
	end
    if !isnothing(τ) 
      Σ[diagind(Σ)] .+= τ^2 
    end
    Σ
end

function prior(K::Integer, Π; covfun, fixedlocs)

	# Sample parameters from the prior distribution
	ρ = rand(Π.ρ, 1, K)
	if :τ ∈ keys(Π)
		τ = rand(Π.τ, 1, K)
		θ = vcat(ρ, τ)
	else 
		τ =  nothing
		θ = ρ
	end

	if fixedlocs
		# Grid over the unit square
		pts = range(0, 1, length = 16)
		S = expandgrid(pts, pts)
		D = pairwise(Euclidean(), S, dims = 1)
	else 
		# Simulate spatial configurations over the unit square
		n = 250 # fixed expected sample size
		λ = rand(Uniform(10, 90), K)
		S = [maternclusterprocess(λ = λ[k], μ = n/λ[k]) for k ∈ 1:K]
		D = pairwise.(Ref(Euclidean()), S, dims = 1)
	end

	# Cholesky factor of covariance matrix
	chols = Folds.map(1:K) do k
		Dₖ = fixedlocs ? D : D[k]
		ρₖ = ρ[k]
		τₖ = isnothing(τ) ? nothing : τ[k]
		Σ = covariancematrix(Dₖ; ρ = ρₖ, τ = τₖ, covfun = covfun)
		cholesky(Symmetric(Σ)).L
	end

	# Convert to Float32 for computational efficiency
	θ = Float32.(θ)

	Parameters(θ, chols, S)
end

function simulate(parameters::Parameters, m::Integer = 1) 
	Folds.map(parameters.chols) do L
		zₖ = simulategaussianprocess(L, m)
		Float32.(zₖ)
	end
end


# ---- Training, validation, and testing sets ----

# Sampled parameters
K = 50000
θ_train = prior(K, Π; covfun = covfun, fixedlocs = fixedlocs)
θ_val   = prior(K ÷ 10, Π; covfun = covfun, fixedlocs = fixedlocs)
θ_test  = prior(1000, Π; covfun = covfun, fixedlocs = fixedlocs)
p = size(θ_test, 1) # number of parameters in the statistical model

# Simulated data sets 
Z_train = simulate(θ_train)
Z_val   = simulate(θ_val)
Z_test = simulate(θ_test)

# Graph version for use in GNN
function reshapeGNN(θ, Z; kwargs...)
	S = θ.S
	if isa(S, Vector)
		g = spatialgraph.(S; kwargs...)
		spatialgraph.(g, Z)
	else 
		g = spatialgraph(S; kwargs...)
		spatialgraph.(Ref(g), Z)
	end
end

g_train = reshapeGNN(θ_train, Z_train; maxmin = false)
g_val = reshapeGNN(θ_val, Z_val; maxmin = false)
g_test = reshapeGNN(θ_test, Z_test; maxmin = false)

g_train_maxmin = reshapeGNN(θ_train, Z_train; maxmin = true)
g_val_maxmin = reshapeGNN(θ_val, Z_val; maxmin = true)
g_test_maxmin = reshapeGNN(θ_test, Z_test; maxmin = true)

# ---- GNN ----

aggr = mean
dₕ = 256  # dimension of final node feature vectors
propagation = GNNChain(
	SpatialGraphConv(1 => 64, w_channels = 16, aggr = aggr),
	SpatialGraphConv(64 => dₕ, aggr = aggr)
	)
readout = GlobalPool(mean); dᵣ = dₕ
ϕ = Chain(Dense(dᵣ, 128, relu), Dense(128, p))

ψ = GNNSummary(propagation, readout)
deepset = DeepSet(ψ, ϕ)
GNN  = PointEstimator(deepset)
GNN2 = PointEstimator(deepcopy(deepset))

GNN = train(GNN, θ_train, θ_val, g_train, g_val)
GNN2 = train(GNN2, θ_train, θ_val, g_train_maxmin, g_val_maxmin)

assessment_GNN = assess(GNN, θ_test, g_test, estimator_name = "GNN: k-nearest", parameter_names = parameter_names)
assessment_GNN2 = assess(GNN2, θ_test, g_test_maxmin, estimator_name = "GNN: maxmin", parameter_names = parameter_names)

## Also consider a GNN that does not account for the spatial locations 
propagation = GNNChain(
	GraphConv(1 => 64, aggr = aggr),
	GraphConv(64 => dₕ, aggr = aggr)
	)
readout = GlobalPool(mean); dᵣ = dₕ
ϕ = Chain(Dense(dᵣ, 128, relu), Dense(128, p))
ψ = GNNSummary(propagation, readout)
deepset = DeepSet(ψ, ϕ)
GNN3 = PointEstimator(deepset)
GNN3 = train(GNN3, θ_train, θ_val, g_train, g_val)
assessment_GNN3 = assess(GNN3, θ_test, g_test, estimator_name = "GNN: k-nearest, non-spatial", parameter_names = parameter_names)


# ---- MAP ----

function MAP(Z::V, ξ) where {T, N, A <: AbstractArray{T, N}, V <: AbstractVector{A}}

	S = ξ.S   # spatial locations 
	D = isa(S, Vector) ? pairwise.(Ref(Euclidean()), S, dims = 1) : pairwise(Euclidean(), S, dims = 1)
	Π = ξ.Π   # prior
	covfun = ξ.covfun   # covariance function

	# Compute MAP for each data set
	θ = Folds.map(eachindex(Z)) do k
		Dₖ = isa(D, Vector) ? D[k] : D
		MAP(Z[k], Dₖ, Π, covfun)
	end

	# Convert to matrix
	θ = reduce(hcat, θ) 

	return θ
end


function MAP(Z::A, D, Π, covfun) where {T, N, A <: AbstractArray{T, N}}

	# Compress the data from a multidimensional array to a matrix
	Z = flatten(Z)
	Z = Float64.(Z)
	Π = [Π...] # convert to array since broadcasting over dictionaries and NamedTuples is reserved

	# Initial estimates
	θ₀ = mean.(Π)

	# Closure that will be minimised
	loss(θ) = nll(θ, Z, D, Π, covfun)

	# Estimate the parameters
	θ = optimize(loss, θ₀, NelderMead()) |> Optim.minimizer

	# During optimisation, we constrained the parameters using the scaled-logistic
	# function; here, we convert to the orginal scale
	θ = scaledlogistic.(θ, Π)

	return θ
end

function nll(θ, Z, D, Π, covfun)

	# Constrain the estimates to be within the prior support
	θ  = scaledlogistic.(θ, Π)

	# compute the negative log-likelihood function
	-logmvnorm(θ, Z, D; covfun = covfun)
end

function logmvnorm(θ, Z, D; kwargs...)
	ρ = θ[1]
	τ = length(θ) == 1 ? nothing : θ[2]
	Σ = covariancematrix(D; τ = τ, ρ = ρ, kwargs...)
	gaussiandensity(Z, Σ; logdensity = true)
end

ξ = (Π = Π, S = θ_test.S, covfun = covfun)
assessment_MAP = assess(MAP, θ_test, Z_test; ξ = ξ, estimator_name = "MAP", parameter_names = parameter_names)

assessment = merge(assessment_MAP, assessment_GNN, assessment_GNN2)

# ---- CNN ----

if fixedlocs
	# Images for CNN 
	reshapeCNN(Z) = reshape.(Z, 16, 16, 1, :)
	image_train = reshapeCNN(Z_train)
	image_val = reshapeCNN(Z_val)
	image_test = reshapeCNN(Z_test)

	ψ = Chain(
		Conv((3, 3), 1 => 32, relu),
		MaxPool((2, 2)),
		Conv((3, 3),  32 => 64, relu),
		MaxPool((2, 2)),
		Flux.flatten
		)
	ϕ = Chain(Dense(256, 128, relu), Dense(128, p))
	CNN = PointEstimator(DeepSet(ψ, ϕ))
	CNN = train(CNN, θ_train, θ_val, image_train, image_val)
	assessment_CNN = assess(CNN, θ_test, image_test, estimator_name = "CNN", parameter_names = parameter_names)
	assessment = merge(assessment, assessment_CNN)
end


# ---- Results ----

img_path = joinpath("img", "GNN_vs_CNN")
mkpath(img_path)
img_name = "$(fixedlocs ? "fixedlocs" : "variablelocs")_$(covfun)_$(noise ? "withnoise" : "withoutnoise")"

num_estimators = length(unique(assessment.df[:, :estimator]))

figure = plot(assessment; grid = true)
save(joinpath(img_path, "$(img_name).png"), figure, px_per_unit = 3, size = (300*num_estimators, 300p))


assessment = merge(assessment_GNN, assessment_GNN3)
assessment.df[:, :estimator] = replace.(assessment.df[:, :estimator], "GNN: k-nearest, non-spatial"=>"GNN: non-spatial")
assessment.df[:, :estimator] = replace.(assessment.df[:, :estimator], "GNN: k-nearest"=>"GNN: spatial")
figure = NeuralEstimators.plot(assessment; grid = false)
num_estimators = length(unique(assessment.df[:, :estimator]))
save(joinpath(img_path, "$(img_name)_spatialweighting.png"), figure, px_per_unit = 3, size = (300+300*p, 300))
CSV.write(joinpath(img_path, "$(img_name)_spatialweighting.csv"), assessment.df)

# R code for generating the final figure:
source("src/plotting.R")
df <- read.csv("img/GNN_vs_CNN/fixedlocs_matern_withnoise_spatialweighting.csv")
p <- length(unique(df$parameter))
figure <- plotestimates(df, parameter_labels = parameter_labels) 
ggsv(figure, file = "fixedlocs_matern_withnoise_spatialweighting", path = "img/GNN_vs_CNN", width = 1+ 3*p, height = 3)

# ---- MCMC (can be really slow, particularly for Matern) ----

# function MCMC(Z::V, ξ) where {T, N, A <: AbstractArray{T, N}, V <: AbstractVector{A}}

# 	D = ξ.D   # distance matrix
# 	Π = ξ.Π   # prior
# 	covfun = ξ.covfun   # covariance function
	
# 	# Compute posterior median for each data set
# 	θ = Folds.map(eachindex(Z)) do k
# 		 MCMC(Z[k], D, Π, covfun)
# 	end

# 	# Convert from vector of vectors to matrix
# 	θ = reduce(hcat, θ)

# 	return θ
# end

# function MCMC(Z::A, D, Π, covfun; nMH = 7000, burn = 2000, thin = 10) where {T, N, A <: AbstractArray{T, N}}

# 	# Compress the data from a multidimensional array to a matrix
# 	Z = flatten(Z)
# 	Z = Float64.(Z)

# 	# initialise MCMC chain
# 	θ₀ = mean.([Π...])
# 	θ = Array{typeof(θ₀)}(undef, nMH)
# 	θ[1] = θ₀

# 	# prior support
# 	Π = [Π...] # convert to array since broadcasting over dictionaries and NamedTuples is reserved

# 	# compute initial likelihood
# 	ℓ_current = logmvnorm(θ₀, Z, D; covfun = covfun)
# 	n_accept = 0
# 	sd_propose = 0.1

# 	for i in 2:nMH

# 		## Propose
# 		# multivariate Gaussian
# 		θ_prop = θ[i-1] .+ sd_propose .* randn(2)

# 		## Accept/Reject
# 		if !all(θ_prop .∈ support.(Π))
# 		  α = 0
# 		else
# 		  ℓ_new = logmvnorm(θ_prop, Z, D; covfun = covfun)
# 		  α = exp(ℓ_new - ℓ_current)
# 	  	end

# 		if rand(1)[1] < α
# 		  # Accept
# 		  θ[i] = θ_prop
# 		  ℓ_current = ℓ_new
# 		  n_accept += 1
# 		else
# 		  ## Reject
# 		  θ[i] = θ[i-1]
# 	  	end

# 		## Monitor acceptance rate
# 		acc_ratio = n_accept / i
# 		#if i % 1000 == 0 println("Sample $i: Acceptance rate: $acc_ratio") end

# 		## If within the burn-in period, adapt acceptance rate
# 		if (i < burn) & (i % 100 == 0)
# 		  if acc_ratio < 0.15
# 			## Decrease proposal variance
# 			sd_propose /= 1.1
# 		  else acc_ratio > 0.4
# 			## Increase proposal variance
# 			sd_propose *= 1.1
# 		  end
# 	  	end
# 	end

# 	# remove burn-in samples and thin the chain
# 	θ = θ[(burn+1):end]
# 	θ = θ[1:thin:end]

# 	# compute marginal medians
# 	θ = reduce(hcat, θ)
# 	θ = median(θ; dims = 2)
# 	θ = vec(θ)

# 	return θ
# end

# θ = θ_test.θ[:, 1]
# Z = Z_test[1]
# D = θ_test.D
# MCMC(Z, D, Π, covfun)

# θ = θ_test.θ[:, 1:2]
# Z = Z_test[1:2]
# ξ = (Π = Π, D = D, covfun = covfun)
# MCMC(Z, ξ)

# assessment_MCMC = assess(MCMC, θ_test[:, 1:100], Z_test[1:100]; ξ = ξ, estimator_name = "MAP", parameter_names = parameter_names)
# assessment = merge(assessment_CNN, assessment_GNN, assessment_MAP, assessment_MCMC)
# figure = plot(assessment)
# save(joinpath(img_path, "$(img_name).png"), figure, px_per_unit = 3, size = (300*4, 300p))



# ---- Unused code for now ----

# Experimental:
# dₕ = 256  # dimension of final node feature vectors
# propagation = GNNChain(
# 	SpatialGraphConv(1 => 64; width = 64),
# 	SpatialGraphConv(64 => dₕ;  width = 64)
# 	)
# readout = GlobalPool(mean); dᵣ = dₕ
# ϕ = Chain(Dense(dᵣ, 256, relu), Dense(256, p))

# Experimental: Skip connections
# dₕ = 256  # dimension of final node feature vectors
# propagation = GNNChain(
# 	GraphSkipConnection(SpatialGraphConv(1 => 64, c = 16)),
# 	GraphSkipConnection(SpatialGraphConv(64 + 1 => dₕ))
# 	)
# readout = GlobalPool(mean); dᵣ = dₕ + 65
# #readout = SpatialPyramidPool(mean); dᵣ = dₕ * 21

# Experimental: SpatialPyramidPool
# dₕ = 64  # dimension of final node feature vectors
# propagation = GNNChain(
# 	SpatialGraphConv(1 => 64, c = 16),
# 	SpatialGraphConv(64 => dₕ)
# 	)
# readout = SpatialPyramidPool(mean); dᵣ = dₕ * 21