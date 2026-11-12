"""
    unitary_overlap(M, W, composite_tensor_arr, raw_ref_op_dag, site_i,
                    dim_left_index, dim_right_index, dim_left_neighbors,
                    dim_right_neighbors; kwargs...)

Computes the unitary overlap (or infidelity/distance measure) between a target tensor
network and a reference operator daggered, after applying a linear operator or
matrix `W` to the active site's composite tensor.

# Arguments
- `M`: Manifold on which `W` resides.
- `W`: Linear operator/matrix applied to the combined site tensor components.
- `composite_tensor_arr`: Array representation of the composite tensor with indices ordered as `combined_index`, followed by `combined_left_neighbors` and `combined_right_neighbors`.
- `raw_ref_op_dag`: Vector of reordered raw tensors representing the reference operator's adjoint chain.
- `site_i`: The site index indicating the starting position of the two-site operator.
- `dim_left_index`: Dimension of the left index on which the isometry acts.
- `dim_right_index`: Dimension of the right index on which the isometry acts.
- `dim_left_neighbors`: Array containing the dimensions of the left neighboring environment indices.
- `dim_right_neighbors`: Array containing the dimensions of the right neighboring environment indices.
- `kwargs...`: Additional keyword arguments passed to underlying optimization or norm functions.

# Returns
- A floating-point value representing `1 - (1 / 2^n) * real(normalized_Frobenius_norm_opt(up_composite_tensor_arr, raw_ref_op_dag, site_i)`, where `n` is the length of `raw_ref_op_dag`
  and `norm` is computed via `normalized_Frobenius_norm_opt`, see [`Disentanglers.normalized_Frobenius_norm_opt`](@ref).
"""
function unitary_overlap(M, W, composite_tensor_arr, raw_ref_op_dag, site_i, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    n = length(raw_ref_op_dag)

    if dim_left_neighbors == [-1]

        # composite_tensor_arr has shape: combined_site, combined_site', right_index

        composite_tensor_arr_r = reshape(composite_tensor_arr, size(composite_tensor_arr)[1], size(composite_tensor_arr)[2] * size(composite_tensor_arr)[3])

        composite_tensor_arr_r = W * composite_tensor_arr_r

        composite_tensor_arr_r = reshape(composite_tensor_arr_r, size(composite_tensor_arr)[1], size(composite_tensor_arr)[2], size(composite_tensor_arr)[3])

        return 1 - (1 / 2.0^n) * real(normalized_Frobenius_norm_opt(composite_tensor_arr_r, raw_ref_op_dag, site_i))

    elseif dim_right_neighbors == [-1]

        # composite_tensor_arr has shape: left_index, combined_site, combined_site'

        composite_tensor_arr_p = permutedims(composite_tensor_arr, (2, 1, 3))

        composite_tensor_arr_rp = reshape(composite_tensor_arr_p, size(composite_tensor_arr)[2], size(composite_tensor_arr)[1] * size(composite_tensor_arr)[3])

        composite_tensor_arr_rp = W * composite_tensor_arr_rp

        composite_tensor_arr_rp = reshape(composite_tensor_arr_rp, size(composite_tensor_arr)[2], size(composite_tensor_arr)[1], size(composite_tensor_arr)[3])

        composite_tensor_arr_rp = permutedims(composite_tensor_arr_rp, (2, 1, 3))

        return 1 - (1 / 2.0^n) * real(normalized_Frobenius_norm_opt(composite_tensor_arr_rp, raw_ref_op_dag, site_i))


    else

        # composite_tensor_arr has shape: left_index, combined_site, combined_site', right_index

        composite_tensor_arr_p = permutedims(composite_tensor_arr, (2, 1, 3, 4))

        composite_tensor_arr_rp = reshape(composite_tensor_arr_p, size(composite_tensor_arr)[2], size(composite_tensor_arr)[1] * size(composite_tensor_arr)[3] * size(composite_tensor_arr)[4])

        composite_tensor_arr_rp = W * composite_tensor_arr_rp

        composite_tensor_arr_rp = reshape(composite_tensor_arr_rp, size(composite_tensor_arr)[2], size(composite_tensor_arr)[1], size(composite_tensor_arr)[3], size(composite_tensor_arr)[4])

        composite_tensor_arr_rp = permutedims(composite_tensor_arr_rp, (2, 1, 3, 4))

        return 1 - (1 / 2.0^n) * real(normalized_Frobenius_norm_opt(composite_tensor_arr_rp, raw_ref_op_dag, site_i))

    end


end

"""
    grad_mat_uo(M, p, composite_tensor_arr, raw_ref_op_dag, site_i,
                dim_left_index, dim_right_index, dim_left_neighbors,
                dim_right_neighbors; kwargs...)

Computes the gradient of the unitary overlap (`unitary_overlap`) with respect to
the parameter/matrix `p` using automatic differentiation (via `DifferentiationInterface.jl`
and the `Zygote.jl` backend).

# Arguments
- `M`: Manifold on which `p` resides.
- `p`: The target parameter array or matrix with respect to which the gradient is computed.
- `composite_tensor`: Array representation of the composite tensor.
- `raw_ref_op_dag`: Vector of reordered raw tensors representing the reference operator's adjoint chain.
- `site_i`: The site index indicating the starting position of the two-site operator.
- `dim_left_index`: Dimension of the left index on which the isometry acts.
- `dim_right_index`: Dimension of the right index on which the isometry acts.
- `dim_left_neighbors`: Array containing the dimensions of the left neighboring environment indices.
- `dim_right_neighbors`: Array containing the dimensions of the right neighboring environment indices.
- `kwargs...`: Additional keyword arguments passed down to `unitary_overlap`.

# Returns
- The computed gradient (`grad`) of the unitary overlap cost function with respect to `p`.

See also: [`isometry_unitary_overlap`](@ref), [`unitary_overlap`](@ref)
"""
function grad_mat_uo(M, p, composite_tensor_arr, raw_ref_op_dag, site_i, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    target_f(p) = unitary_overlap(M, p, composite_tensor_arr, raw_ref_op_dag, site_i, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    backend = AutoZygote()

    grad = DifferentiationInterface.gradient(target_f, backend, p)

    return grad

end

"""
    isometry_unitary_overlap(composite_object, raw_ref_op_dag, site_i,
                             left_index, right_index, left_neighbors, right_neighbors;
                             gradient_free_solver = false, manifold = Stiefel,
                             manifold_field = ℂ, opt = quasi_Newton,
                             initial_p0 = "RS", p0_val = nothing, kwargs...)

Optimizes an isometry/unitary transformation over a manifold to minimize the distance
with a reference operator adjoint for a composite tensor network site.

!!! note
    This function is designed to be used in conjunction with [`apply_optimized_isometry`](@ref) in order
    to return to the original form. See examples [`Compressing the MPO`](@ref DisentanglingMPO).

# Arguments
- `composite_object`: Composite `ITensor` object on which the isometry acts.
- `raw_ref_op_dag`: Vector of reordered raw tensors representing the reference operator's adjoint chain.
- `site_i`: The site index indicating the starting position of the two-site operator.
- `dim_left_index`: Dimension of the left index on which the isometry acts.
- `dim_right_index`: Dimension of the right index on which the isometry acts.
- `dim_left_neighbors`: Array containing the dimensions of the left neighboring environment indices.
- `dim_right_neighbors`: Array containing the dimensions of the right neighboring environment indices.

# Keyword Arguments
- `gradient_free_solver = false`: Boolean flag to toggle gradient-free optimization solvers.
- `manifold = Stiefel`: Manifold constructor for optimization (e.g., `Stiefel`, can also be `Unitary`, `Grassmannian`).
- `manifold_field = ℂ`: Number field type over which the manifold is defined (defaults to Complex, ℂ but can also be Real, ℝ field).
- `opt`: The optimization solver function (default: `quasi_Newton`).
- `initial_p0 = "RS"`: Strategy for initial point generation (`"RS"`: `Random Split`, `"RE"`: `Random Entangled`, `"I"`: `Identity`, or `"C"`: `Custom`).
- `p0_val = nothing`: Custom initial matrix on the manifold value if `initial_p0 = "C"`.
- `kwargs...`: Additional keyword arguments passed down to the optimizer and objective routines.

# Returns
- `u_kij`: Optimized isometry as an `ITensor` with primed and unprimed combined indices.
- `res`: Final solver state or result object.
- `initial_objective_val`: The initial objective function value evaluated at the identity matrix.
- `final_objective_val`: The final objective function value evaluated using the optimized solver result.
- `elapsed_time`: Total execution time for the optimization process in seconds.
"""
function isometry_unitary_overlap(composite_object, raw_ref_op_dag, site_i, left_index, right_index, left_neighbors, right_neighbors; gradient_free_solver = false, manifold = Stiefel, manifold_field = ℂ, opt = quasi_Newton, initial_p0 = "RS", p0_val = nothing, kwargs...)

    combined_tensor = combiner(left_index, right_index; tags = "comb")
    combined_index = combinedind(combined_tensor)
    composite_tensor = composite_object * combined_tensor

    combined_tensor_p = combiner(left_index', right_index'; tags = "combp")
    combined_index_p = combinedind(combined_tensor_p)
    composite_tensor = composite_tensor * combined_tensor_p

    # permute to get into the form of left_index, combined_index, combined_index', right_index
    ordering_indices = Index[]
    if length(left_neighbors) != 0
        append!(ordering_indices, left_neighbors)
    end

    append!(ordering_indices, [combined_index])
    append!(ordering_indices, [combined_index_p])

    if length(right_neighbors) != 0
        append!(ordering_indices, right_neighbors)
    end

    # permute with left neighbor at the start
    composite_tensor = permute(composite_tensor, ordering_indices)

    if length(left_neighbors) != 0
        dim_left_neighbors = [dim(ln) for ln in left_neighbors]
    else
        dim_left_neighbors = [-1]
    end

    if length(right_neighbors) != 0
        dim_right_neighbors = [dim(rn) for rn in right_neighbors]
    else
        dim_right_neighbors = [-1]
    end

    dim_left_index = dim(left_index)
    dim_right_index = dim(right_index)
    composite_tensor_arr = Array(composite_tensor.tensor)

    # call the objective function
    f(M, p) = unitary_overlap(M, p, composite_tensor_arr, raw_ref_op_dag, site_i, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    # define the gradient of the objective function
    grad_f(M, p) = project(M, p, grad_mat_uo(M, p, composite_tensor_arr, raw_ref_op_dag, site_i, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...))

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

export unitary_overlap, isometry_unitary_overlap
