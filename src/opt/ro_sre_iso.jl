"""
    second_renyi_entropy(M, W, composite_tensor_arr, dim_left_index, dim_right_index,
                         dim_left_neighbors, dim_right_neighbors; kwargs...)

Compute the second-Rényi entropy (``n = 2``) associated with the virtual bond shared
by adjacent tensors forming a composite tensor object after applying an isometry matrix `W`
on manifold `M`.

## Arguments
- `M`: Manifold on which the isometry matrix is defined.
- `W`: Isometry matrix applied to the reshaped composite tensor.
- `composite_tensor_arr`: Array representation of the composite tensor with indices ordered as `combined_index`, followed by `combined_left_neighbors` and `combined_right_neighbors`.
- `dim_left_index`: Dimension of the left index on which the isometry acts.
- `dim_right_index`: Dimension of the right index on which the isometry acts.
- `dim_left_neighbors`: Array containing the dimensions of the left neighboring environment indices.
- `dim_right_neighbors`: Array containing the dimensions of the right neighboring environment indices.

## Keywords
- `kwargs...`: Additional keyword arguments (reserved for future extensions).

## Returns
- The computed second-order Rényi entropy value (``- \\log \\text{Tr}(\\rho^2)``), equivalent to `alpha_renyi_entropy` at `alpha=2`.

See also: [`alpha_renyi_entropy`](@ref)
"""
function second_renyi_entropy(M, W, composite_tensor_arr, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    dim_combined_index = dim_left_index * dim_right_index

    # permute this to have k_ij, chi_i, s_i, s_j, chi_j -> this is the input! (k_ij is the leading index)
    # should receive a permuted tensor with combined_index followed by the rest of indices as input!

    # reshape to get the matrix with structure: combined_index × rest of the indices
    total_env_dim = prod(dim_left_neighbors) * prod(dim_right_neighbors)
    composite_tensor_arr = reshape(composite_tensor_arr, dim_combined_index, total_env_dim)

    #### get two parts: isometry multiplied by a composite tensor and the dagger version
    # now act with W (the isometry) from the left!
    composite_tensor_arr_n = W * composite_tensor_arr
    composite_tensor_arr_n_dag = adjoint(composite_tensor_arr) * adjoint(W)

    #### now split each of the two parts to have left and right index
    composite_tensor_arr_r = reshape(composite_tensor_arr_n, dim_left_index, dim_right_index, total_env_dim)
    composite_tensor_arr_r_dag = reshape(composite_tensor_arr_n_dag, total_env_dim, dim_left_index, dim_right_index)

    #### permute the indices, move the right index to the right
    composite_tensor_arr_p = permutedims(composite_tensor_arr_r, (1, 3, 2))
    composite_tensor_arr_p_dag = permutedims(composite_tensor_arr_r_dag, (2, 1, 3))


    #### put all the left indices together and right indices together
    composite_tensor_arr_p = reshape(composite_tensor_arr_p, dim_left_index * prod(dim_left_neighbors), prod(dim_right_neighbors) * dim_right_index)
    composite_tensor_arr_p_dag = reshape(composite_tensor_arr_p_dag, dim_left_index * prod(dim_left_neighbors), prod(dim_right_neighbors) * dim_right_index)

    #### multiply the right indices together and leave the left as is!
    composite_tensor_arr_p = permutedims(composite_tensor_arr_p, (2, 1))
    rho_contract = composite_tensor_arr_p_dag * composite_tensor_arr_p

    #### contract the left indices
    rho_sq_contract = rho_contract * rho_contract

    return -log(real(tr(rho_sq_contract)))

end


"""
    grad_mat_sre(M, p, composite_tensor, left_index, right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

Compute the gradient of the second-Rényi entropy objective function with respect to
the isometry parameter matrix `p` on the manifold `M` using automatic differentiation (Zygote).

## Arguments
- `M`: Manifold on which `p` resides.
- `p`: The target parameter array or matrix with respect to which the gradient is computed.
- `composite_tensor`: Array representation of the composite tensor.
- `left_index`: Dimension of the left index on which the isometry acts.
- `right_index`: Dimension of the right index on which the isometry acts.
- `dim_left_neighbors`: Dimensions of the left neighboring environment indices.
- `dim_right_neighbors`: Dimensions of the right neighboring environment indices.

## Keywords
- `kwargs...`: Additional keyword arguments forwarded to [`alpha_renyi_entropy`](@ref).

## Returns
- The evaluated derivative matrix corresponding to the objective.

See also: [`isometry_ro_sre`](@ref), [`second_renyi_entropy`](@ref)
"""
function grad_mat_sre(M, p, composite_tensor, left_index, right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    target_f(p) = second_renyi_entropy(M, p, composite_tensor, left_index, right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    backend = AutoZygote()

    grad = DifferentiationInterface.gradient(target_f, backend, p)

    return grad

end

"""
    isometry_ro_sre(composite_object, left_index, right_index, left_neighbors, right_neighbors;
                    gradient_free_solver = false, manifold = Stiefel, manifold_field = ℂ,
                    opt = quasi_Newton, initial_p0 = "RS", p0_val = nothing, kwargs...)

Return the optimized isometry that minimizes the second-order Rényi entropy of a composite
tensor network object across the specified cut.

!!! note
    This function is designed to be used in conjunction with [`apply_optimized_isometry`](@ref) in order
    to return to the original form. See examples [`Disentangling the MPS`](@ref DisentanglingMPS), [`Compressing the MPO`](@ref DisentanglingMPO), [`Optimizing the LPDO`](@ref DisentanglingLPDO).

## Arguments
- `composite_object`: Composite `ITensor` object on which the isometry acts.
- `left_index`: Left index out of the two indices on which the isometry acts.
- `right_index`: Right index out of the two indices on which the isometry acts.
- `left_neighbors`: Array of indices that end up on the left after the SVD cut.
- `right_neighbors`: Array of indices that end up on the right after the SVD cut.

## Keywords
- `gradient_free_solver = false`: Boolean flag to toggle gradient-free optimization solvers.
- `manifold = Stiefel`: Manifold constructor for optimization (e.g., `Stiefel`, can also be `Unitary`, `Grassmannian`).
- `manifold_field = ℂ`: Number field type over which the manifold is defined (defaults to Complex, ℂ but can also be Real, ℝ field).
- `opt`: The optimization solver function (default: `quasi_Newton`).
- `initial_p0 = "RS"`: Strategy for initial point generation (`"RS"`: `Random Split`, `"RE"`: `Random Entangled`, `"I"`: `Identity`, or `"C"`: `Custom`).
- `p0_val = nothing`: Custom initial matrix on the manifold value if `initial_p0 = "C"`.
- `kwargs...`: Additional keyword arguments passed to the underlying solver or entropy evaluation functions.

## Returns
- `u_kij`: Optimized isometry as an `ITensor` with primed and unprimed combined indices.
- `res`: Final solver state or result object.
- `initial_objective_val`: Initial objective value before optimization.
- `final_objective_val`: The final objective function value evaluated using the optimized solver result.
- `elapsed_time`: Total execution time for the optimization process in seconds.



See also: [`isometry_ro_are`](@ref), [`isometry_ts`](@ref)
"""
function isometry_ro_sre(composite_object, left_index, right_index, left_neighbors, right_neighbors; gradient_free_solver = false, manifold = Stiefel, manifold_field = ℂ, opt = quasi_Newton, initial_p0 = "RS", p0_val = nothing, kwargs...)

    ################## TODO: Include options to inflate the left and right index

    ##################

    combined_tensor = combiner(left_index, right_index; tags = "comb")
    combined_index = combinedind(combined_tensor)
    composite_tensor = composite_object * combined_tensor

    # permute to get the combined_index to the beginning!
    # ordering is crucial to compute the entropy at the right cut!
    ordering_indices = [combined_index]
    if length(left_neighbors) != 0
        append!(ordering_indices, left_neighbors)
    end
    if length(right_neighbors) != 0
        append!(ordering_indices, right_neighbors)
    end

    # permute with combined index at the start
    composite_tensor = permute(composite_tensor, ordering_indices)

    if length(left_neighbors) != 0
        dim_left_neighbors = [dim(ln) for ln in left_neighbors]
    else
        dim_left_neighbors = [1]
    end

    if length(right_neighbors) != 0
        dim_right_neighbors = [dim(rn) for rn in right_neighbors]
    else
        dim_right_neighbors = [1]
    end

    # pass everything as scalars so that we can GPUify them later!
    dim_left_index = dim(left_index)
    dim_right_index = dim(right_index)
    composite_tensor_arr = Array(composite_tensor.tensor)


    # call the objective function
    f(M, p) = second_renyi_entropy(M, p, composite_tensor_arr, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    # define the gradient of the objective function
    grad_f(M, p) = project(M, p, grad_mat_sre(M, p, composite_tensor_arr, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...))

    M = manifold(dim(combined_index), dim(combined_index), manifold_field)

    # initialize the optimizer
    if initial_p0 == "RS"
        M1 = manifold(dim(left_index), dim(left_index), manifold_field)
        M2 = manifold(dim(right_index), dim(right_index), manifold_field)
        p0 = kron(rand(M1), rand(M2))
    elseif initial_p0 == "RE"
        p0 = rand(M)
    elseif initial_p0 == "I"
        p0 = Matrix((1.0 + 0.0im)I, dim(combined_index), dim(combined_index))
    elseif initial_p0 == "C" && p0_val != nothing
        p0 = p0_val
    end

    p_id = Matrix((1.0 + 0.0im)I, dim(combined_index), dim(combined_index))
    initial_objective_val = f(M, p_id)

    if gradient_free_solver == true

        start_time = time()

        res = opt(M, f; return_state = true, kwargs...)

        elapsed_time = time() - start_time

    elseif opt == stochastic_gradient_descent

        start_time = time()

        res = opt(M, grad_f, p0; cost = f, return_state = true, kwargs...)

        elapsed_time = time() - start_time

    else


        start_time = time()

        res = opt(M, f, grad_f, p0; return_state = true, kwargs...)

        elapsed_time = time() - start_time

    end

    final_objective_val = f(M, get_solver_result(res))


    u_kij = ITensor(get_solver_result(res), combined_index', combined_index)


    return u_kij, res, initial_objective_val, final_objective_val, elapsed_time

end

export second_renyi_entropy, isometry_ro_sre
