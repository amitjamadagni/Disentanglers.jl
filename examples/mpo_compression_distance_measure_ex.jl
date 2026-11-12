using ITensors
using ITensorMPS
using Disentanglers
using Manifolds
using Manopt

# using Random -> to generate the seed

"""
    benchmark_mpo_distance_measure_opt(opt_function, r_mpo, nlayers, r_seed; kwargs...)

Benchmarks an optimization routine for MPO distance measures (such as unitary overlap compression/disentangling)
across multiple layers of a Matrix Product Operator (MPO). Sweeps through odd and even site pairs,
applies the optimization function, updates the MPO via SVD, and tracks entanglement entropy changes,
objective function values, link dimensions, and execution times.

# Arguments
- `opt_function`: The optimization function to apply per site-pair (e.g., `isometry_unitary_overlap`).
- `r_mpo`: The target Matrix Product Operator (`MPO`) to be optimized/benchmarked.
- `nlayers::Int`: Maximum number of optimization layers to perform.
- `r_seed::Int`: Random seed for reproducibility.
- `kwargs...`: Additional keyword arguments passed down to `opt_function`.

# Keyword Arguments
- `kwargs...`: Additional keyword arguments passed directly to `opt_function`.

# Returns
- `layers::Int`: The actual number of layers executed before convergence or completion.
- `sf_vn_ent_arr`: Array of post-update von Neumann entanglement entropy values.
- `delta_vn_ent_arr`: Array of changes in von Neumann entanglement entropy (pre minus post update) per step.
- `obj_func_val_arr`: Array of final objective function values from each optimization step.
- `linkdims_arr`: Array tracking average link dimensions across layers.
- `time_arr`: Array of elapsed execution times recorded per site-pair optimization.

# Examples
```julia
flayers, final_sv_ent_arr, delta_vn_ent, fobj_func_val_arr, link_dims, time_arr = Disentanglers.benchmark_mpo_distance_measure_opt(
    isometry_unitary_overlap, ham_evo_mpo, nlayers, seed; opt=solver)
```
"""
function benchmark_mpo_distance_measure_opt(opt_function, r_mpo, nlayers, r_seed; kwargs...)

    N = length(r_mpo)

    sites = [noprime(uniqueind(siteinds(r_mpo)[_k])) for _k in 1:N]

    layers = 0

    # store change in no. of singular values, change in entropy and time!
    delta_vn_ent_arr = Float64[]
    sf_vn_ent_arr = Float64[]
    obj_func_val_arr = Float64[]
    time_arr = Float64[]
    linkdims_arr = Float64[]

    link_dims = linkdims(r_mpo)
    link_dims_avg = sum(link_dims) / length(link_dims)
    push!(linkdims_arr, link_dims_avg)

    Random.seed!(r_seed)

    while layers < nlayers

        if linkdims(r_mpo) == [1 for l in 1:(length(sites) - 1)]

            break

        else

            layers = layers + 1

            for odd_site in 1:2:(N - 1)

                if linkdim(r_mpo, odd_site) != 1

                    orthogonalize!(r_mpo, odd_site)

                    left_index = sites[odd_site]
                    right_index = sites[odd_site + 1]

                    if odd_site == 1
                        left_neighbors = Index[]
                        left_neighbors_n = [left_index']
                    else
                        left_neighbors = [linkind(r_mpo, odd_site - 1)]
                        left_neighbors_n = [linkind(r_mpo, odd_site - 1), left_index']
                    end

                    if odd_site != N - 1
                        right_neighbors = [linkind(r_mpo, odd_site + 1)]
                        right_neighbors_n = [linkind(r_mpo, odd_site + 1), right_index']

                    else
                        right_neighbors = Index[]
                        right_neighbors_n = Index[right_index']
                    end

                    ## construct the nearest neighbor tensor
                    two_site_tensor = r_mpo[odd_site] * r_mpo[odd_site + 1]

                    svd_left_indices = vcat(left_neighbors_n, left_index)

                    _, S_i, _ = svd(two_site_tensor, svd_left_indices, cutoff = 1.0e-8)

                    raw_ref_op_dag = permute_op_and_return_raw_data(r_mpo, true)

                    ## perform the optimization on the nearest neighbors
                    opt_iso, _, iov, fov, elapsed_time = opt_function(two_site_tensor, raw_ref_op_dag, odd_site, left_index, right_index, left_neighbors, right_neighbors; kwargs...)

                    # apply the optimized isometry on the clubbed composite object
                    two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors_n, right_neighbors_n)

                    # post update SVD
                    U, S_f, V = svd(two_site_tensor_update, svd_left_indices, righttags = "Link,l=" * string(odd_site), cutoff = 1.0e-8)

                    s_i_arr = diag(Array(S_i.tensor))
                    s_f_arr = diag(Array(S_f.tensor))


                    r_mpo[odd_site] = U * S_f
                    r_mpo[odd_site + 1] = V

                    push!(time_arr, elapsed_time)

                    vn_ent_pre = alpha_renyi_cutoff(diag(Array(S_i.tensor)), 1, 0.0; renormalize = true)
                    vn_ent_post = alpha_renyi_cutoff(diag(Array(S_f.tensor)), 1, 0.0; renormalize = true)

                    push!(sf_vn_ent_arr, vn_ent_post)

                    push!(delta_vn_ent_arr, vn_ent_pre - vn_ent_post)

                    push!(obj_func_val_arr, fov)


                else


                end

            end


            for even_site in 2:2:(N - 2)

                if linkdim(r_mpo, even_site) != 1

                    orthogonalize!(r_mpo, even_site)

                    left_index = sites[even_site]
                    right_index = sites[even_site + 1]

                    left_neighbors = [linkind(r_mpo, even_site - 1)]
                    right_neighbors = [linkind(r_mpo, even_site + 1)]

                    left_neighbors_n = [linkind(r_mpo, even_site - 1), left_index']
                    right_neighbors_n = [linkind(r_mpo, even_site + 1), right_index']

                    ## construct the nearest neighbor tensor
                    two_site_tensor = r_mpo[even_site] * r_mpo[even_site + 1]

                    svd_left_indices = vcat(left_neighbors_n, left_index)

                    _, S_i, _ = svd(two_site_tensor, svd_left_indices, cutoff = 1.0e-8)

                    raw_ref_op_dag = permute_op_and_return_raw_data(r_mpo, true)

                    ## perform the optimization on the nearest neighbors
                    opt_iso, _, iov, fov, elapsed_time = opt_function(two_site_tensor, raw_ref_op_dag, even_site, left_index, right_index, left_neighbors, right_neighbors; kwargs...)

                    # apply the optimized isometry on the clubbed composite object
                    two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors_n, right_neighbors_n)

                    U, S_f, V = svd(two_site_tensor_update, svd_left_indices, righttags = "Link,l=" * string(even_site), cutoff = 1.0e-8)

                    s_i_arr = diag(Array(S_i.tensor))
                    s_f_arr = diag(Array(S_f.tensor))

                    r_mpo[even_site] = U * S_f
                    r_mpo[even_site + 1] = V

                    push!(time_arr, elapsed_time)

                    vn_ent_pre = alpha_renyi_cutoff(diag(Array(S_i.tensor)), 1, 0.0; renormalize = true)
                    vn_ent_post = alpha_renyi_cutoff(diag(Array(S_f.tensor)), 1, 0.0; renormalize = true)

                    push!(sf_vn_ent_arr, vn_ent_post)

                    push!(delta_vn_ent_arr, vn_ent_pre - vn_ent_post)

                    push!(obj_func_val_arr, fov)


                else

                end

            end

        end

        link_dims = linkdims(r_mpo)
        link_dims_avg = sum(link_dims) / length(link_dims)
        push!(linkdims_arr, link_dims_avg)

    end

    return layers, sf_vn_ent_arr, delta_vn_ent_arr, obj_func_val_arr, linkdims_arr, time_arr

end
