using ITensors
using ITensorMPS
using Disentanglers
using Manifolds
using Manopt

# using Random -> to generate the seed

"""
    benchmark_mps_opt(opt_function, r_mps, nlayers, ro_opt, r_seed; kwargs...)

Disentangling Matrix Product State (MPS) optimization layers using a specified optimization
function (`opt_function`).

# Arguments
- `opt_function`: The optimization routine used for disentangling two-site tensors.
- `r_mps::MPS`: The initial Matrix Product State to be optimized/disentangled.
- `nlayers::Integer`: The maximum number of sweeping layers to perform.
- `ro_opt::Bool`: Flag indicating whether Riemannian optimization (`true`) or standard methods (`false`) are used.
- `r_seed::Integer`: Random seed for the solver (mainly used for reproducibility).

# Keyword Arguments
- `kwargs...`: Additional keyword arguments passed directly to `opt_function`.

# Returns
- `layers::Int`: The actual number of layers executed before convergence or function termination.
- `sf_vn_ent_arr::Vector{Float64}`: Von-Neumann entropy values after update application.
- `delta_vn_ent_arr::Vector{Float64}`: Change in Von Neumann entropy before and after updates.
- `delta_nsv_arr::Vector{Float64}`: Change in the number of singular values.
- `linkdims_arr::Vector{Float64}`: Average link dimensions of the MPS per layer.
- `time_arr::Vector{Float64}`: Elapsed execution times for the disentangling steps.

# Examples
```julia
# Setting the options (replace the seed and solver appropriately before calling)
flayers, final_sv_ent_arr, delta_vn_ent, delta_nsv, link_dims, time_arr = Disentanglers.benchmark_mps_opt(isometry_ro_are, mps_gs, nlayers, true, seed; opt = solver, gradient_free_solver = true)
flayers, final_sv_ent_arr, delta_vn_ent, delta_nsv, link_dims, time_arr = Disentanglers.benchmark_mps_opt(isometry_ro_sre, mps_gs, nlayers, true, seed; opt = solver)
```
"""
function benchmark_mps_opt(opt_function, r_mps, nlayers, ro_opt, r_seed; kwargs...)

    N = length(r_mps)

    sites = siteinds(r_mps)

    layers = 0

    # store change in no. of singular values, change in entropy and time!
    delta_nsv_arr = Float64[]
    delta_vn_ent_arr = Float64[]
    sf_vn_ent_arr = Float64[]
    time_arr = Float64[]
    linkdims_arr = Float64[]

    link_dims = linkdims(r_mps)
    link_dims_avg = sum(link_dims) / length(link_dims)
    push!(linkdims_arr, link_dims_avg)

    Random.seed!(r_seed)

    while layers < nlayers

        if linkdims(r_mps) == [1 for l in 1:(length(sites) - 1)]

            break

        else

            layers = layers + 1

            for odd_site in 1:2:(N - 1)

                if linkdim(r_mps, odd_site) != 1

                    orthogonalize!(r_mps, odd_site)


                    left_index = sites[odd_site]
                    right_index = sites[odd_site + 1]

                    if odd_site == 1
                        left_neighbors = []
                    else
                        left_neighbors = [linkind(r_mps, odd_site - 1)]
                    end

                    if odd_site != N - 1
                        right_neighbors = [linkind(r_mps, odd_site + 1)]
                    else
                        right_neighbors = []
                    end

                    ## construct the nearest neighbor tensor
                    two_site_tensor = r_mps[odd_site] * r_mps[odd_site + 1]

                    svd_left_indices = vcat(left_neighbors, left_index)

                    # pre update SVD
                    _, S_i, _ = svd(two_site_tensor, svd_left_indices, cutoff = 1.0e-8)

                    ## perform the optimization on the nearest neighbors
                    if ro_opt == true
                        opt_iso, _, _, _, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; kwargs...)
                    else
                        _, opt_iso, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; kwargs...)
                    end

                    # apply the optimized isometry on the clubbed composite object
                    two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors, right_neighbors)

                    # post update SVD
                    U, S_f, V = svd(two_site_tensor_update, svd_left_indices, righttags = "Link,l=" * string(odd_site), cutoff = 1.0e-8)

                    s_i_arr = diag(Array(S_i.tensor))
                    s_f_arr = diag(Array(S_f.tensor))
                    if (length(s_i_arr) == length(s_f_arr)) && all(isapprox(s_i_arr, s_f_arr))

                        if ro_opt == true
                            opt_iso, _, _, _, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; initial_p0 = "RE", retraction_method = QRRetraction(), kwargs...)
                        else
                            _, opt_iso, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; initial_p0 = "RE", kwargs...)
                        end

                        # apply the optimized isometry on the clubbed composite object
                        two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors, right_neighbors)

                        # post update SVD
                        U, S_f, V = svd(two_site_tensor_update, svd_left_indices, righttags = "Link,l=" * string(odd_site), cutoff = 1.0e-8)

                        r_mps[odd_site] = U * S_f
                        r_mps[odd_site + 1] = V

                        normalize!(r_mps)

                        if elapsed_time == nothing

                            push!(time_arr, 0)

                        else

                            push!(time_arr, elapsed_time)

                        end

                    else

                        r_mps[odd_site] = U * S_f
                        r_mps[odd_site + 1] = V

                        normalize!(r_mps)

                        if elapsed_time == nothing

                            push!(time_arr, 0)

                        else

                            push!(time_arr, elapsed_time)

                        end

                    end

                    # record the difference in vn entropy before and after
                    vn_ent_pre = alpha_renyi_cutoff(diag(Array(S_i.tensor)), 1, 0.0)
                    vn_ent_post = alpha_renyi_cutoff(diag(Array(S_f.tensor)), 1, 0.0)

                    push!(sf_vn_ent_arr, vn_ent_post)

                    push!(delta_vn_ent_arr, vn_ent_pre - vn_ent_post)

                    # record the difference in singular values before and after
                    push!(delta_nsv_arr, length(diag(Array(S_i.tensor))) - length(diag(Array(S_f.tensor))))

                else


                end

            end


            for even_site in 2:2:(N - 2)

                if linkdim(r_mps, even_site) != 1

                    orthogonalize!(r_mps, even_site)

                    left_index = sites[even_site]
                    right_index = sites[even_site + 1]

                    left_neighbors = [linkind(r_mps, even_site - 1)]
                    right_neighbors = [linkind(r_mps, even_site + 1)]

                    ## construct the nearest neighbor tensor
                    two_site_tensor = r_mps[even_site] * r_mps[even_site + 1]

                    svd_left_indices = vcat(left_neighbors, left_index)

                    # pre update SVD
                    _, S_i, _ = svd(two_site_tensor, svd_left_indices, cutoff = 1.0e-8)

                    ## perform the optimization on the nearest neighbors
                    if ro_opt == true
                        opt_iso, _, _, _, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; kwargs...)
                    else
                        _, opt_iso, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; kwargs...)
                    end

                    # apply the optimized isometry on the clubbed composite object
                    two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors, right_neighbors)

                    # post update SVD
                    U, S_f, V = svd(two_site_tensor_update, svd_left_indices, righttags = "Link,l=" * string(even_site), cutoff = 1.0e-8)

                    s_i_arr = diag(Array(S_i.tensor))
                    s_f_arr = diag(Array(S_f.tensor))

                    if (length(s_i_arr) == length(s_f_arr)) && all(isapprox(s_i_arr, s_f_arr))

                        if ro_opt == true
                            opt_iso, _, _, _, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; initial_p0 = "RE", retraction_method = QRRetraction(), kwargs...)
                        else
                            _, opt_iso, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; initial_p0 = "RE", kwargs...)
                        end

                        # apply the optimized isometry on the clubbed composite object
                        two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors, right_neighbors)

                        # post update SVD
                        U, S_f, V = svd(two_site_tensor_update, svd_left_indices, righttags = "Link,l=" * string(even_site), cutoff = 1.0e-8)

                        r_mps[even_site] = U * S_f
                        r_mps[even_site + 1] = V

                        normalize!(r_mps)

                        if elapsed_time == nothing

                            push!(time_arr, 0)

                        else

                            push!(time_arr, elapsed_time)

                        end

                    else

                        r_mps[even_site] = U * S_f
                        r_mps[even_site + 1] = V

                        normalize!(r_mps)

                        if elapsed_time == nothing

                            push!(time_arr, 0)

                        else

                            push!(time_arr, elapsed_time)

                        end

                    end

                    # record the difference in vn entropy before and after
                    vn_ent_pre = alpha_renyi_cutoff(diag(Array(S_i.tensor)), 1, 0.0)
                    vn_ent_post = alpha_renyi_cutoff(diag(Array(S_f.tensor)), 1, 0.0)

                    push!(sf_vn_ent_arr, vn_ent_post)

                    push!(delta_vn_ent_arr, vn_ent_pre - vn_ent_post)

                    # record the difference in singular values before and after
                    push!(delta_nsv_arr, length(diag(Array(S_i.tensor))) - length(diag(Array(S_f.tensor))))

                else

                end

            end

        end

        link_dims = linkdims(r_mps)
        link_dims_avg = sum(link_dims) / length(link_dims)
        push!(linkdims_arr, link_dims_avg)

    end

    return layers, sf_vn_ent_arr, delta_vn_ent_arr, delta_nsv_arr, linkdims_arr, time_arr

end

### Example calls:
### -------------

"""
    benchmark_ro_are_opt(init_mps, nlayers; kwargs...)

Default call for Riemannian disentanglers with α-Rényi entropy as the objective function, (`isometry_ro_are`).

# Arguments
- `init_mps`: The initial Matrix Product State (MPS) to disentangle/optimize.
- `nlayers::Integer`: The maximum number of sweeping layers.
- `kwargs...`: Additional keyword arguments passed to `benchmark_mps_opt`.

# Returns
- `layers::Int`: The actual number of layers executed before completion.
- `sf_vn_ent_arr::Vector{Float64}`: Von-Neumann entropy values after updates.
- `delta_vn_ent_arr::Vector{Float64}`: Change in von-Neumann entropy before and after updates.
- `delta_nsv_arr::Vector{Float64}`: Change in the number of singular values.
- `linkdims_arr::Vector{Float64}`: Average link dimensions of the MPS per layer.
- `time_arr::Vector{Float64}`: Elapsed execution times for the optimization steps.
"""
benchmark_ro_are_opt(init_mps, nlayers; kwargs...) = benchmark_mps_opt(isometry_ro_are, init_mps, nlayers, true, rand(); kwargs...)


"""
    benchmark_ro_sre_opt(init_mps, nlayers; kwargs...)

Default call for Riemannian disentanglers with second-Rényi entropy as the objective function, (`isometry_ro_sre`).

# Arguments
- `init_mps`: The initial Matrix Product State (MPS) to optimize.
- `nlayers::Integer`: The maximum number of sweeping layers.
- `kwargs...`: Additional keyword arguments passed to `benchmark_mps_opt`.

# Returns
- `layers::Int`: The actual number of layers executed before completion.
- `sf_vn_ent_arr::Vector{Float64}`: Von-Neumann entropy values after updates.
- `delta_vn_ent_arr::Vector{Float64}`: Change in von Neumann entropy before and after updates.
- `delta_nsv_arr::Vector{Float64}`: Change in the number of singular values.
- `linkdims_arr::Vector{Float64}`: Average link dimensions of the MPS per layer.
- `time_arr::Vector{Float64}`: Elapsed execution times for the optimization steps.
"""
benchmark_ro_sre_opt(init_mps, nlayers; kwargs...) = benchmark_mps_opt(isometry_ro_sre, init_mps, nlayers, true, rand(); kwargs...)


"""
    benchmark_ts_opt(init_mps, nlayers; kwargs...)

Default call for the Tensor sketching algorithm (`isometry_ts`) with non-Riemannian options (`ro_opt = false`).

# Arguments
- `init_mps`: The initial Matrix Product State (MPS) to optimize.
- `nlayers::Integer`: The maximum number of sweeping layers.
- `kwargs...`: Additional keyword arguments passed to `benchmark_mps_opt`.

# Returns
- `layers::Int`: The actual number of layers executed before completion.
- `sf_vn_ent_arr::Vector{Float64}`: Von-Neumann entropy values after updates.
- `delta_vn_ent_arr::Vector{Float64}`: Change in von Neumann entropy before and after updates.
- `delta_nsv_arr::Vector{Float64}`: Change in the number of singular values.
- `linkdims_arr::Vector{Float64}`: Average link dimensions of the MPS per layer.
- `time_arr::Vector{Float64}`: Elapsed execution times for the optimization steps.

# Examples
```julia
flayers, final_sv_ent_arr, delta_vn_ent, delta_nsv, link_dims, time_arr = benchmark_ts_opt(mps_gs, nlayers)
```
"""
benchmark_ts_opt(init_mps, nlayers; kwargs...) = benchmark_mps_opt(isometry_ts, init_mps, nlayers, false, rand(); kwargs...)

"""
    benchmark_ev_opt(init_mps, nlayers; kwargs...)

Default call for the Evenbly-Vidal algorithm (`isometry_ev`) with non-Riemannian options (`ro_opt = false`).

# Arguments
- `init_mps`: The initial Matrix Product State (MPS) to optimize.
- `nlayers::Integer`: The maximum number of sweeping layers.
- `kwargs...`: Additional keyword arguments passed to `benchmark_mps_opt`.

# Returns
- `layers::Int`: The actual number of layers executed before completion.
- `sf_vn_ent_arr::Vector{Float64}`: Von Neumann entropy values after updates.
- `delta_vn_ent_arr::Vector{Float64}`: Change in von Neumann entropy before and after updates.
- `delta_nsv_arr::Vector{Float64}`: Change in the number of singular values.
- `linkdims_arr::Vector{Float64}`: Average link dimensions of the MPS per layer.
- `time_arr::Vector{Float64}`: Elapsed execution times for the optimization steps.

# Examples
```julia
flayers, final_sv_ent_arr, delta_vn_ent, delta_nsv, link_dims, time_arr = benchmark_ev_opt(mps_gs, nlayers)
```
"""
benchmark_ev_opt(init_mps, nlayers; kwargs...) = benchmark_mps_opt(isometry_ev, init_mps, nlayers, false, rand(); kwargs...)
