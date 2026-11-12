# [Integrating new objective functions](@id Integratingnewobjfuncs)

We now present ways in which new objective functions can be integrated into the package. It demands defining the following functions:

- Objective function,
- Derivative of the objective function,
- Optimization function that calls the objective and its derivative

As an example, we integrate the (in)fidelity between two wavefunctions, ψ and ϕ ``1 - |<ψ|ϕ>|^2`` as the objective function:

We define the objective function as follows:
```julia
function overlap(M, W, composite_tensor_arr, psi_arr, i, eta_dag_arr, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; Λ_cutoff = 1.0e-8, kwargs...)

    dim_combined_index = dim_left_index * dim_right_index

    # permute this to have k_ij, chi_i, s_i, s_j, chi_j -> this is the input!
    # should receive a permuted tensor with combined_index followed by the rest of indices as input!
    # composite_tensor_arr = Array(composite_tensor.tensor)

    # reshape to get the matrix with structure: combined_index × rest of the indices
    total_env_dim = prod(dim_left_neighbors) * prod(dim_right_neighbors)
    composite_tensor_arr = reshape(composite_tensor_arr, dim_combined_index, total_env_dim)

    # now act with W (the isometry) from the left!
    composite_tensor_arr_n = W * composite_tensor_arr

    # now reshape this to split the combined index -> combined index = left index × right index
    composite_tensor_arr_r = reshape(composite_tensor_arr_n, dim_left_index, dim_right_index, total_env_dim)

    # move the right index to the right
    composite_tensor_arr_p = permutedims(composite_tensor_arr_r, (1, 3, 2))

    # reshape with left indices and its neighbors on one side, right indices and its neighbors on the other
    # composite_tensor_arr_p = reshape(composite_tensor_arr_p, dim_left_index*prod(dim_left_neighbors), prod(dim_right_neighbors)*dim_right_index)

    # custom inner requires an array with left_link = sites = right_link
    if i == 1

        composite_tensor_arr_p = reshape(composite_tensor_arr_p, dim_left_index, prod(dim_right_neighbors), dim_right_index)
        composite_tensor_arr_p = permutedims(composite_tensor_arr_p, (1, 3, 2))
        return 1 - abs(inner_custom(composite_tensor_arr_p, i, psi_arr, eta_dag_arr))^2

    elseif i == length(psi_arr) - 1

        composite_tensor_arr_p = reshape(composite_tensor_arr_p, dim_left_index, prod(dim_left_neighbors), dim_right_index)
        composite_tensor_arr_p = permutedims(composite_tensor_arr_p, (2, 1, 3))
        return 1 - abs(inner_custom(composite_tensor_arr_p, i, psi_arr, eta_dag_arr))^2

    else

        composite_tensor_arr_p = reshape(composite_tensor_arr_p, dim_left_index, prod(dim_left_neighbors), prod(dim_right_neighbors), dim_right_index)
        composite_tensor_arr_p = permutedims(composite_tensor_arr_p, (2, 1, 4, 3))
        return 1 - abs(inner_custom(composite_tensor_arr_p, i, psi_arr, eta_dag_arr))^2

    end

end
```

Define its numerical derivative as follows:
```julia
function grad_mat_overlap(M, p, composite_tensor, psi_arr, i, eta_dag_arr, left_index, right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    target_f(p) = overlap(M, p, composite_tensor, psi_arr, i, eta_dag_arr, left_index, right_index, dim_left_neighbors, dim_right_neighbors; kwargs...)

    backend = AutoZygote()

    grad = DifferentiationInterface.gradient(target_f, backend, p)

    return grad

end
```

Finally integrate the above functions into the optimization function:
```julia
function isometry_overlap(composite_object, psi_arr, i, eta_dag_arr, left_index, right_index, left_neighbors, right_neighbors; λ_cutoff = 1.0e-8, manifold = Stiefel, manifold_field = ℂ, opt = quasi_Newton, initial_p0 = "RS", kwargs...)

    ##################

    combined_tensor = combiner(left_index, right_index; tags = "comb")
    combined_index = combinedind(combined_tensor)
    composite_tensor = composite_object * combined_tensor

    # permute to get the combined_index to the beginning!
    # ordering is crucial to compute the right entropy at the right cut!
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

    # PRE OPTIMIZATION SETUP DONE!

    # pass everything as scalars so that we can GPUify them later!
    dim_left_index = dim(left_index)
    dim_right_index = dim(right_index)
    composite_tensor_arr = Array(composite_tensor.tensor)

    # call the objective function
    f(M, p) = overlap(M, p, composite_tensor_arr, psi_arr, i, eta_dag_arr, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; Λ_cutoff = λ_cutoff, kwargs...)

    grad_f(M, p) = project(M, p, grad_mat_overlap(M, p, composite_tensor_arr, psi_arr, i, eta_dag_arr, dim_left_index, dim_right_index, dim_left_neighbors, dim_right_neighbors; kwargs...))

    M = manifold(dim(combined_index), dim(combined_index), manifold_field)

    # initialize the optimizer
    if initial_p0 == "RS"
        M1 = manifold(dim(left_index), dim(left_index), manifold_field)
        M2 = manifold(dim(right_index), dim(right_index), manifold_field)
        p0 = kron(rand(M1), rand(M2))
    elseif initial_p0 == "RE"
        p0 = rand(M)
    elseif initial_p0 == "I"
        # id_m = Matrix((1.0+0.0im)I, dim(left_index), dim(right_index))
        # p0 = kron(id_m, id_m)
        p0 = Matrix((1.0 + 0.0im)I, dim(combined_index), dim(combined_index))
    end

    res = opt(M, f, grad_f, p0; return_state = true, kwargs...)

    u_kij = ITensor(get_solver_result(res), combined_index', combined_index)

    return u_kij, res

end
```
