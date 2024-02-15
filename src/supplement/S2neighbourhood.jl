# ------------------------------------------------------------------------------
#  Experiment: Comparing definitions for the neighbourhood of a node
#  - Disc of fixed spatial radius
#  - k-nearest neighbours for fixed k
#  - Random-k neighbours within a disc of fixed spatial radius
#  - Maxmin ordering
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

model = joinpath("GP", "nuSigmaFixed")
m = 1
using NeuralEstimators
using NeuralEstimatorsGNN
using BenchmarkTools
using DataFrames
using GraphNeuralNetworks
using CSV

include(joinpath(pwd(), "src", model, "model.jl"))
include(joinpath(pwd(), "src", "architecture.jl"))

path = "intermediates/supplement/neighbours"
if !isdir(path) mkpath(path) end

# Size of the training, validation, and test sets
K_train = 10_000
K_val   = K_train ÷ 2
if quick
	K_train = K_train ÷ 100
	K_val   = K_val   ÷ 100
end
K_test = K_val ÷ 2

p = ξ.p

# For uniformly sampled locations on the unit square, the probability that a
# point falls within a circle of radius r << 1 centred away from the boundary is
# simply πr². So, on average, we expect nπr² neighbours for each node. Use this
# information to choose k in a way that makes for a relatively fair comparison.
n = 250  # sample size
r = 0.15 # disc radius
# k = ceil(Int, n*π*r^2)
k = 10


# ---- adjacencymatrix ----

import NeuralEstimators: adjacencymatrix
export adjacencymatrix

using Random: shuffle
using Distances
using InvertedIndices
using NearestNeighbors
using StatsBase: sample
using SparseArrays

function adjacencymatrix(M::Mat, r::F, k::Integer) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert k > 0
	@assert r > 0

	I = Int64[]
	J = Int64[]
	V = Float64[]
	n = size(M, 1)
	m = size(M, 2)

	for i ∈ 1:n
		sᵢ = M[i, :]
		kᵢ = 0
		iter = shuffle(collect(1:n)) # shuffle to prevent weighting observations based on their ordering in M

		for j ∈ iter

			if m == n # square matrix, so assume M is a distance matrix
				dᵢⱼ = M[i, j]
			else
				sⱼ  = M[j, :]
				dᵢⱼ = norm(sᵢ - sⱼ)
			end

			if dᵢⱼ <= r
				push!(I, i)
				push!(J, j)
				push!(V, dᵢⱼ)
				kᵢ += 1
			end
			if kᵢ == k break end
		end

	end

	A = sparse(I,J,V,n,n)


	return A
end
adjacencymatrix(M::Mat, k::Integer, r::F) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat} = adjacencymatrix(M, r, k)

function adjacencymatrix(M::Mat, k::Integer; maxmin::Bool = false, moralise::Bool = false) where Mat <: AbstractMatrix{T} where T

	@assert k > 0

	I = Int64[]
	J = Int64[]
	V = Float64[]
	n = size(M, 1)
	m = size(M, 2)

	if m == n # square matrix, so assume M is a distance matrix
		D = M
	else      # otherwise, M is a matrix of spatial locations
		S = M
	end

	if k >= n # more neighbours than observations: return a dense adjacency matrix
		if m != n
			D = pairwise(Euclidean(), S')
		end
		A = sparse(D)
	elseif !maxmin
		k += 1 # each location neighbours itself, so increase k by 1
		for i ∈ 1:n

			if m == n
				d = D[i, :]
			else
				# Compute distances between sᵢ and all other locations
				d = colwise(Euclidean(), S', S[i, :])
			end

			# Find the neighbours of s
			j, v = findneighbours(d, k)

			push!(I, repeat([i], inner = k)...)
			push!(J, j...)
			push!(V, v...)
		end
		A = sparse(I,J,V,n,n)
	else
		@assert m != n "`adjacencymatrix` with maxmin-ordering requires a matrix of spatial locations, not a distance matrix"
		ord     = ordermaxmin(S)          # calculate ordering
		Sord    = S[ord, :]               # re-order locations
		NNarray = findorderednn(Sord, k)  # find k nearest neighbours/"parents"
		R = builddag(NNarray)             # build DAG
		A = moralise ?  R' * R : R        # moralise

		# Add distances to A
		# NB This is inefficient, especially for large n; only optimise
		#    if we find that this approach works well
		D = pairwise(Euclidean(), Sord')
		I, J, V = findnz(A)
		indices = collect(zip(I,J))
		indices = CartesianIndex.(indices)
		A.nzval .= D[indices]

		# "unorder" back to the original ordering
		# Sanity check: Sord[sortperm(ord), :] == S
		# Sanity check: D[sortperm(ord), sortperm(ord)] == pairwise(Euclidean(), S')
		A = A[sortperm(ord), sortperm(ord)]
	end

	return A
end

function adjacencymatrix(M::Mat, r::F) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert r > 0

	n = size(M, 1)
	m = size(M, 2)

	if m == n # square matrix, so assume M is a distance matrix, D:

		D = M
		# bit-matrix specifying which locations are d-neighbours
		A = D .< r

		# replace non-zero elements of A with the corresponding distance in D
		indices = copy(A)
		A = convert(Matrix{T}, A)
		A[indices] = D[indices]

		# convert to sparse matrix
		A = sparse(A)
	else

		S = M

		I = Int64[]
		J = Int64[]
		V = Float64[]
		for i ∈ 1:n

			#  We don't need to compute all distances, we can
			#  immediately throw away some values if the difference between
			#  any of their marginal coordinates is greater than r. Just
			#  computing the distances seems faster though.
			# a₁ = sᵢ[1] - r
			# b₁ = sᵢ[1] + r
			# a₂ = sᵢ[2] - r
			# b₂ = sᵢ[2] + r
			# a₁ .< S[:, 1] .< b₁
			# a₂ .< S[:, 2] .< b₂

			# Compute distances between s and all other locations
			s = S[i, :]
			d = colwise(Euclidean(), S', s)

			# Find the r-neighbours of s
			j = d .< r
			j = findall(j)
			push!(I, repeat([i], inner = length(j))...)
			push!(J, j...)
			push!(V, d[j]...)

			# Alternative using NearestNeighbors (https://github.com/KristofferC/NearestNeighbors.jl)
			# Didn't find this to be much faster, so sticking with my
			# implementation so that I don't need to add a package dependency.
			# tree = BallTree(S') # this is done once only, outside of the loop
			# j = inrange(tree, sᵢ, r, true)
			# S_subset = S[:, j]
			# d = colwise(Euclidean(), S_subset, sᵢ)
			# push!(I, repeat([i], inner = length(j))...)
			# push!(J, j...)
			# push!(V, d...)
		end
		A = sparse(I,J,V,n,n)
	end

	return A
end

function findneighbours(d, k::Integer)
	V = partialsort(d, 1:k)
	J = [findfirst(v .== d) for v ∈ V]
    return J, V
end

function getknn(S, s, k; args...)
  tree = KDTree(S; args...)
  nn_index, nn_dist = knn(tree, s, k, true)
  nn_index = hcat(nn_index...) |> permutedims # nn_index = stackarrays(nn_index, merge = false)'
  nn_dist  = hcat(nn_dist...)  |> permutedims # nn_dist  = stackarrays(nn_dist, merge = false)'
  nn_index, nn_dist
end

function ordermaxmin_fast(S)

  # get number of locs
  n = size(S, 1)
  k = isqrt(n)
  # k is number of neighbors to search over
  # get the past and future nearest neighbors
  NNall = getknn(S', S', k)[1]
  # pick a random ordering
  index_in_position = [sample(1:n, n, replace = false)..., repeat([missing],1*n)...]
  position_of_index = sortperm(index_in_position[1:n])
  # loop over the first n/4 locations
  # move an index to the end if it is a
  # near neighbor of a previous location
  curlen = n
  nmoved = 0
  for j ∈ 2:2n
	nneigh = round(min(k, n /(j-nmoved+1)))
    nneigh = Int(nneigh)
    neighbors = NNall[index_in_position[j], 1:nneigh]
    if minimum(skipmissing(position_of_index[neighbors])) < j
      nmoved += 1
      curlen += 1
      position_of_index[ index_in_position[j] ] = curlen
      rassign(index_in_position, curlen, index_in_position[j])
      index_in_position[j] = missing
  	end
  end
  ord = collect(skipmissing(index_in_position))

  return ord
end

rowMins(X) = vec(mapslices(minimum, X, dims = 2))
colMeans(X) = vec(mapslices(mean, X, dims = 1))
function ordermaxmin_slow(S)
	n = size(S, 1)
	D = pairwise(Euclidean(), S')
	## Vecchia sequence based on max-min ordering: start with most central location
  	vecchia_seq = [argmin(D[argmin(colMeans(D)), :])]
  	for j in 2:n
    	vecchia_seq_new = (1:n)[Not(vecchia_seq)][argmax(rowMins(D[Not(vecchia_seq), vecchia_seq, :]))]
		rassign(vecchia_seq, j, vecchia_seq_new)
	end
  return vecchia_seq
end

ordermaxmin(S) = size(S, 1) > 200 ? ordermaxmin_fast(S) : ordermaxmin_slow(S)


function rassign(v::AbstractVector, index::Integer, x)
	@assert index > 0
	if index <= length(v)
		v[index] = x
	elseif index == length(v)+1
		push!(v, x)
	else
		v = [v..., fill(missing, index - length(v) - 1)..., x]
	end
	return v
end

function findorderednnbrute(S, k::Integer)
  # find the k+1 nearest neighbors to S[j,] in S[1:j,]
  # by convention, this includes S[j,], which is distance 0
  n = size(S, 1)
  k = min(k,n-1)
  NNarray = Matrix{Union{Integer, Missing}}(missing, n, k+1)
  for j ∈ 1:n
	d = colwise(Euclidean(), S[1:j, :]', S[j, :])
    NNarray[j, 1:min(k+1,j)] = sortperm(d)[1:min(k+1,j)]
  end
  return NNarray
end

function findorderednn(S, k::Integer)

  # number of locations
  n = size(S, 1)
  k = min(k,n-1)
  mult = 2

  # to store the nearest neighbor indices
  NNarray = Matrix{Union{Integer, Missing}}(missing, n, k+1)

  # find neighbours of first mult*k+1 locations by brute force
  maxval = min( mult*k + 1, n )
  NNarray[1:maxval, :] = findorderednnbrute(S[1:maxval, :],k)

  query_inds = min( maxval+1, n):n
  data_inds = 1:n
  ksearch = k
  while length(query_inds) > 0
    ksearch = min(maximum(query_inds), 2ksearch)
    data_inds = 1:min(maximum(query_inds), n)
	NN = getknn(S[data_inds, :]', S[query_inds, :]', ksearch)[1]

    less_than_l = hcat([NN[l, :] .<= query_inds[l] for l ∈ 1:size(NN, 1)]...) |> permutedims
	sum_less_than_l = vec(mapslices(sum, less_than_l, dims = 2))
    ind_less_than_l = findall(sum_less_than_l .>= k+1)
	NN_k = hcat([NN[l,:][less_than_l[l,:]][1:(k+1)] for l ∈ ind_less_than_l]...) |> permutedims
    NNarray[query_inds[ind_less_than_l], :] = NN_k

    query_inds = query_inds[Not(ind_less_than_l)]
  end

  return NNarray
end

function builddag(NNarray)
  n, k = size(NNarray)
  I = [1]
  J = [1]
  V = Float64[1.0]
  for j in 2:n
    i = NNarray[j, :]
    i = collect(skipmissing(i))
    push!(J, repeat([j], length(i))...)
    push!(I, i...)
	push!(V, repeat([1], length(i))...)
  end
  R = sparse(I,J,V,n,n)
  return R
end

# n=100
# S = rand(n, 2)
# k=5
# ord = ordermaxmin(S)              # calculate maxmin ordering
# Sord = S[ord, :];                 # reorder locations
# NNarray = findorderednn(Sord, k)  # find k nearest neighbours/"parents"
# R = builddag(NNarray)             # build the DAG
# Q = R' * R                        # moralise

import NeuralEstimatorsGNN: modifyneighbourhood
function modifyneighbourhood(θ::Parameters, k::Integer; kwargs...)

	S = θ.locations
	if !(typeof(S) <: AbstractVector) S = [S] end

	A = adjacencymatrix.(S, k; kwargs...)
	graphs = GNNGraph.(A)

	Parameters(θ.θ, S, graphs, θ.chols, θ.chol_pointer, θ.loc_pointer)
end


# ---- Testing with small sample sizes ----

# using NeuralEstimatorsGNN: ordermaxmin, getknn, rassign
#
# n = 20
# seed!(3)
# θ = Parameters(1, ξ, n, cluster_process = cluster_process)
#
# seed!(3)
# modifyneighbourhood(θ, k; maxmin = true, moralise = false)
# S = θ.locations[1]
#
# seed!(3)
# ordermaxmin(S)

# ---- initialise the estimators ----

seed!(1)
gnn1 = gnnarchitecture(p)
gnn2 = deepcopy(gnn1)
gnn3 = deepcopy(gnn1)
gnn4 = deepcopy(gnn1)

# ---- Training ----

cluster_process = true
epochs_per_Z_refresh = 3
epochs = quick ? 2 : 300
stopping_epochs = 10000 # very large number to "turn off" early stopping

# Sample parameter vectors
seed!(1)
θ_val   = Parameters(K_val,   ξ, n, J = 5, cluster_process = cluster_process)
θ_train = Parameters(K_train, ξ, n, J = 5, cluster_process = cluster_process)

@info "Training with a disc of fixed radius"
θ̃_val   = modifyneighbourhood(θ_val, r)
θ̃_train = modifyneighbourhood(θ_train, r)
train(gnn1, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "fixedradius"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh, stopping_epochs = stopping_epochs)

@info "Training with k-nearest neighbours with k=$k"
θ̃_val   = modifyneighbourhood(θ_val, k)
θ̃_train = modifyneighbourhood(θ_train, k)
train(gnn2, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "knearest"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh, stopping_epochs = stopping_epochs)

@info "Training with a random set of k=$k neighbours selected within a disc of fixed spatial radius"
θ̃_val   = modifyneighbourhood(θ_val, r, k)
θ̃_train = modifyneighbourhood(θ_train, r, k)
train(gnn3, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "combined"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh, stopping_epochs = stopping_epochs)

@info "Training with maxmin ordering with k=$k neighbours"
θ̃_val   = modifyneighbourhood(θ_val, k;   maxmin = true, moralise = false)
θ̃_train = modifyneighbourhood(θ_train, k; maxmin = true, moralise = false)
train(gnn4, θ̃_train, θ̃_val, simulate, m = m, savepath = joinpath(path, "maxmin"), epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh, stopping_epochs = stopping_epochs)


# ---- Load the trained estimators ----

Flux.loadparams!(gnn1,  loadbestweights(joinpath(path, "fixedradius")))
Flux.loadparams!(gnn2,  loadbestweights(joinpath(path, "knearest")))
Flux.loadparams!(gnn3,  loadbestweights(joinpath(path, "combined")))
Flux.loadparams!(gnn4,  loadbestweights(joinpath(path, "maxmin")))

# ---- Assess the estimators ----

function assessestimators(n, ξ, K::Integer)
	println("	Assessing estimators with n = $n...")

	# test parameter vectors for estimating the risk function
	seed!(1)
	θ = Parameters(K, ξ, n, cluster_process = cluster_process)

	# Estimator trained with a fixed radius
	θ̃ = modifyneighbourhood(θ, r)
	seed!(1); Z = simulate(θ̃, m)
	assessment = assess(
		[gnn1], θ̃, Z;
		estimator_names = ["fixedradius"],
		parameter_names = ξ.parameter_names,
		verbose = false
	)

	# Estimators trained with k-nearest neighbours for fixed k
	θ̃ = modifyneighbourhood(θ, k)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn2], θ̃, Z;
		estimator_names = ["knearest"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))

	# Estimator trained with a random set of k neighbours selected within a disc of fixed spatial radius
	θ̃ = modifyneighbourhood(θ, r, k)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn3], θ̃, Z;
		estimator_names = ["combined"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))

	# Estimators trained with maxmin ordering
	θ̃ = modifyneighbourhood(θ, k; maxmin = true, moralise = false)
	seed!(1); Z = simulate(θ̃, m)
	assessment = merge(assessment, assess(
		[gnn4], θ̃, Z;
		estimator_names = ["maxmin"],
		parameter_names = ξ.parameter_names,
		verbose = false
	))

	# Add sample size information
	assessment.df[:, :n] .= n

	return assessment
end

test_n = [30, 60, 100, 200, 300, 500, 750, 1000]
assessment = [assessestimators(n, ξ, K_test) for n ∈ test_n]
assessment = merge(assessment...)
CSV.write(joinpath(path, "estimates.csv"), assessment.df)
CSV.write(joinpath(path, "runtime.csv"), assessment.runtime)


# ---- Accurately assess the run-time for a single data set ----

@info "Assessing the run-time for a single data set for each estimator and each sample size n ∈ $(test_n)..."

# just use one gnn since the architectures are exactly the same
gnn  = gnn1 |> gpu

function testruntime(n, ξ)

    # Simulate locations and data
	seed!(1)
    S = rand(n, 2)
	D = pairwise(Euclidean(), S, S, dims = 1)
  	ξ = (ξ..., S = S, D = D) # update ξ to contain the new distance matrix D (needed for simulation and ML estimation)
  	θ = Parameters(1, ξ)
  	Z = simulate(θ, m; convert_to_graph = false)

  	seed!(1)
  	θ = Parameters(1, ξ, n, cluster_process = cluster_process)

  	# Fixed radius
  	θ̃ = modifyneighbourhood(θ, r)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn1 = @belapsed gnn($Z)

  	# k-nearest neighbours
    θ̃ = modifyneighbourhood(θ, k)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn2 = @belapsed gnn($Z)

  	# combined
    θ̃ = modifyneighbourhood(θ, r, k)
	Z = simulate(θ̃, m)|> gpu
  	t_gnn3 = @belapsed gnn($Z)

	# maxmin
	θ̃ = modifyneighbourhood(θ, k; maxmin = true, moralise = false)
	Z = simulate(θ̃, m)|> gpu
	t_gnn4 = @belapsed gnn($Z)

  	# Store the run times as a data frame
  	DataFrame(
	time = [t_gnn1, t_gnn2, t_gnn3, t_gnn4],
	estimator = ["fixedradius", "knearest", "combined", "maxmin"],
	n = n)
  end

test_n = [32, 64, 128, 256, 512, 1024, 2048, 4096]
seed!(1)
times = []
for n ∈ test_n
	t = testruntime(n, ξ)
	push!(times, t)
end
times = vcat(times...)

CSV.write(joinpath(path, "runtime_singledataset.csv"), times)


# ---- Sensitivity analysis with respect to k ----

for k ∈ [5, 10, 15, 20, 25, 30]

	@info "Training GNN with maxmin ordering and k=$k"

	savepath = joinpath(path, "maxmin_k$k")

	global θ_val   = modifyneighbourhood(θ_val, k; maxmin = true, moralise = false)
	global θ_train = modifyneighbourhood(θ_train, k; maxmin = true, moralise = false)

	seed!(1)
	gnn = gnnarchitecture(p)
	train(gnn, θ_train, θ_val, simulate, m = m, savepath = savepath,
		  epochs = epochs, epochs_per_Z_refresh = epochs_per_Z_refresh,
		  stopping_epochs = stopping_epochs)
end


assessments = []
for n ∈ test_n
  @info "Assessing the estimators with sample size n=$n"
  seed!(1)
  θ_test   = Parameters(K_test, ξ, n, J = J)
  θ_single = Parameters(1, ξ, n, J = 1)
  for k ∈ [5, 10, 15, 20, 25, 30]

    @info "Assessing the estimator with k=$k"

    gnn = gnnarchitecture(p)
  	loadpath = joinpath(path, "maxmin_k$k")
  	Flux.loadparams!(gnn, loadbestweights(loadpath))

    θ_test   = modifyneighbourhood(θ_test, k; maxmin = true, moralise = false)
    θ_single = modifyneighbourhood(θ_single, k; maxmin = true, moralise = false)

  	# Assess the estimator's accuracy
  	seed!(1)
  	Z_test = simulate(θ_test, m)
  	assessment = assess(
  		[gnn], θ_test, Z_test;
  		estimator_names = ["gnn_k$k"],
  		parameter_names = ξ.parameter_names,
  		verbose = false
  	)
  	assessment.df[:, :k] .= k
  	assessment.df[:, :n] .= n

  	# Accurately assess the inference time for a single data set
    gnn = gnn |> gpu
    Z_single = simulate(θ_single, m) |> gpu
    t = @belapsed $gnn($Z_single)
    assessment.df[:, :inference_time] .= t

    push!(assessments, assessment)
  end
end
assessment = merge(assessments...)
CSV.write(joinpath(path, "k_vs_n.csv"), assessment.df)
