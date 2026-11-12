using ITensors
using ITensorMPS
using Disentanglers
using Manifolds
using Manopt
using LinearAlgebra

# using Random -> to generate the seed

function κ_ind_finder(Q::ITensor, s_ind::Int64)
    for i in inds(Q)
        if contains(string(tags(i)), "Mixtures,κ_" * string(s_ind))
            return i
        end
    end
    return
end

function phys_ind_finder(Q::Union{MPS, MPO}, s_ind::Int64)
    for i in inds(Q[s_ind])
        if contains(string(tags(i)), "S=1/2,Site") # && plev(i) == 0
            return i
        end
    end
    return
end

function HSIP(A, B)

    # trace over mixture indicies
    HS = A[1] * delta(κ_ind_finder(A[1], 1), κ_ind_finder(A[1], 1)') * dag(A[1])'
    HS *= B[1]'' * delta(κ_ind_finder(B[1], 1)'', κ_ind_finder(B[1], 1)''') * dag(B[1])'''

    # then the physical indicies
    HS *= delta(phys_ind_finder(A, 1)', phys_ind_finder(B, 1)'')
    HS *= delta(phys_ind_finder(A, 1), phys_ind_finder(B, 1)''')

    for j in 2:length(A)
        # and get the χs by shared indicies
        HS *= A[j] * delta(κ_ind_finder(A[j], j), κ_ind_finder(A[j], j)') * dag(A[j])'
        HS *= B[j]'' * delta(κ_ind_finder(B[j], j)'', κ_ind_finder(B[j], j)''') * dag(B[j])'''
        HS *= delta(phys_ind_finder(A, j)', phys_ind_finder(B, j)'')
        HS *= delta(phys_ind_finder(A, j), phys_ind_finder(B, j)''')
    end
    ## println(HS.tensor[1])
    return real(HS.tensor[1])
end


function F_HS(A, B)
    return HSIP(A, B) / max(HSIP(A, A), HSIP(B, B))
end

function LPDO_Id(sites)

    N = length(sites)

    mpo_op = MPO(N)

    for s in 1:N

        op_id = op("Id", sites[s])

        U, S, V = svd(op_id, sites[s]', lefttags = "Mixtures,κ_" * string(s))

        mpo_op[s] = sqrt.(S) * V / sqrt(2)

    end

    return mpo_op

end

function singular_discard_renorm(svals, δ; rno = false)

    rev_svals = reverse(svals)

    sval_sum = 0.0

    cutoff_ind = nothing

    for (ind_s, sval) in enumerate(rev_svals)

        sval_sum = sval_sum + sval^2

        if sval_sum >= δ

            cutoff_ind = ind_s

            break

        end

    end

    retained_svals = svals[1:(length(svals) - (cutoff_ind - 1))]

    discard_svals = rev_svals[1:(cutoff_ind - 1)]

    if rno == false
        ## println(" RNO OFF")
        renorm_val = sqrt(norm(svals, 2)^2 - norm(discard_svals, 2)^2)
    elseif rno == true
        ## println(" RNO ON")
        renorm_val = sqrt(1 - norm(discard_svals, 2)^2)
    end

    svals_renorm = retained_svals / renorm_val

    return svals_renorm, norm(discard_svals, 2)

end

function chi_opt(ρ, i, j, Λ_chi, clubbed_object; c_obj = nothing, rno = false)

    if clubbed_object == false

        mprho_ij = ρ[i] * ρ[j]

    else

        mprho_ij = c_obj

    end

    s_i = siteind(ρ, i)
    s_j = siteind(ρ, j)

    k_i_mprho = κ_ind_finder(mprho_ij, i)
    k_j_mprho = κ_ind_finder(mprho_ij, j)

    if i != 1
        chi_i = commoninds(mprho_ij, ρ[i - 1])[1]
    end

    if j != length(ρ)
        chi_j = commoninds(mprho_ij, ρ[j + 1])[1]
    end

    # always work in the array data structure,
    # independent of the ITensor, two steps:
    # A. Permute the indices,
    # B. Reshape to perform SVD

    if i == 1

        mprho_ij_p = permute(mprho_ij, s_i, k_i_mprho, s_j, chi_j, k_j_mprho)
        mprho_ij_p_arr = Array(mprho_ij_p.tensor)
        mprho_ij_re = reshape(mprho_ij_p_arr, dim(s_i) * dim(k_i_mprho), dim(s_j) * dim(chi_j) * dim(k_j_mprho))

    end

    if j == length(ρ)

        mprho_ij_p = permute(mprho_ij, s_i, chi_i, k_i_mprho, s_j, k_j_mprho)
        mprho_ij_p_arr = Array(mprho_ij_p.tensor)
        mprho_ij_re = reshape(mprho_ij_p_arr, dim(s_i) * dim(chi_i) * dim(k_i_mprho), dim(s_j) * dim(k_j_mprho))

    end

    if i != 1 && j != length(ρ)

        mprho_ij_p = permute(mprho_ij, s_i, chi_i, k_i_mprho, s_j, chi_j, k_j_mprho)
        mprho_ij_p_arr = Array(mprho_ij_p.tensor)
        mprho_ij_re = reshape(mprho_ij_p_arr, dim(s_i) * dim(chi_i) * dim(k_i_mprho), dim(s_j) * dim(chi_j) * dim(k_j_mprho))

    end

    U1, S1, V1 = LinearAlgebra.svd(mprho_ij_re)
    V1t = adjoint(V1)

    S1_tilde, discarded_norm = singular_discard_renorm(S1, Λ_chi; rno = rno)
    svd_arr_1 = S1_tilde

    U1_tilde = U1[:, 1:length(S1_tilde)]
    US_tilde = U1_tilde * diagm(S1_tilde)

    V_tilde = V1t[1:length(S1_tilde), :]

    link_index = Index(length(S1_tilde), "Link,l=" * string(i))

    if i == 1

        US_tilde_re = reshape(US_tilde, dim(s_i), dim(k_i_mprho), length(S1_tilde))
        V_tilde_re = reshape(V_tilde, length(S1_tilde), dim(s_j), dim(chi_j), dim(k_j_mprho))

        US_tilde_it = ITensor(US_tilde_re, s_i, k_i_mprho, link_index)
        V_tilde_it = ITensor(V_tilde_re, link_index, s_j, chi_j, k_j_mprho)

    end

    if j == length(ρ)

        US_tilde_re = reshape(US_tilde, dim(s_i), dim(chi_i), dim(k_i_mprho), length(S1_tilde))
        V_tilde_re = reshape(V_tilde, length(S1_tilde), dim(s_j), dim(k_j_mprho))

        US_tilde_it = ITensor(US_tilde_re, s_i, chi_i, k_i_mprho, link_index)
        V_tilde_it = ITensor(V_tilde_re, link_index, s_j, k_j_mprho)

    end

    if i != 1 && j != length(ρ)

        US_tilde_re = reshape(US_tilde, dim(s_i), dim(chi_i), dim(k_i_mprho), length(S1_tilde))
        V_tilde_re = reshape(V_tilde, length(S1_tilde), dim(s_j), dim(chi_j), dim(k_j_mprho))

        US_tilde_it = ITensor(US_tilde_re, s_i, chi_i, k_i_mprho, link_index)
        V_tilde_it = ITensor(V_tilde_re, link_index, s_j, chi_j, k_j_mprho)

    end


    return US_tilde_it, V_tilde_it, svd_arr_1, discarded_norm

end

"""
    benchmark_lpdo_opt(opt_function, r_lpdo, nlayers, r_seed, opt_cutoff; kwargs...)

Optimize the representation of Locally Purified Density Operator (LPDO) using a specified optimization
function (`opt_function`) and sweeping routines. The isometry acts on the κ-indices that encode the mixture
correlations. The main goal is to prune the coherent correlations, encoded by the χ-indices using the κ-indices.

# Arguments
- `opt_function`: The optimization routine used for updating two-site LPDO tensors.
- `r_lpdo`: The initial LPDO whose representation is to be optimized by disentangling.
- `nlayers::Integer`: The maximum number of sweeping layers to perform.
- `r_seed::Integer`: Random seed for the solver (mainly used for reproducibility).
- `opt_cutoff`: Cutoff threshold for pruning χ (see J. Math. Phys. 67, 072201 (2026) for setting the cutoff).

# Keyword Arguments
- `kwargs...`: Additional keyword arguments passed directly to `opt_function`.

# Returns
- `layers::Int`: The actual number of layers executed before convergence or completion.
- `sf_vn_ent_arr::Vector{Float64}`: Von Neumann entropy values after updates.
- `delta_vn_ent_arr::Vector{Float64}`: Change in Von Neumann entropy before and after updates.
- `delta_nsv_arr::Vector{Float64}`: Change in the number of active singular/spectrum values.
- `linkdims_arr::Vector{Float64}`: Average link dimensions of the LPDO per layer.
- `hsip_arr::Vector{Float64}`: Hilbert-Schmidt inner product / fidelity values measured against the initial state.
- `time_arr::Vector{Float64}`: Elapsed execution times for the optimization steps.

# Examples
```julia
# set the seed and cutoff values

## using the α-Rényi entropy as the objective function with some choice of the solver
layers, sf_ent, delta_ent, delta_nsv, link_dims, hsip, times = Disentanglers.benchmark_lpdo_opt(
    isometry_ro_are, lpdo_init, nlayers, seed, cutoff; opt = solver)

## using the second-Rényi entropy as the objective function with some choice of the solver
layers, sf_ent, delta_ent, delta_nsv, link_dims, hsip, times = Disentanglers.benchmark_lpdo_opt(
	isometry_ro_sre, lpdo_init, nlayers, seed, 1.0e-8; opt = solver)
```
"""
function benchmark_lpdo_opt(opt_function, r_lpdo, nlayers, r_seed, opt_cutoff; kwargs...)

    N = length(r_lpdo)

    r_lpdo_dc = deepcopy(r_lpdo)

    sites = siteinds(r_lpdo)

    layers = 0

    # store change in no. of singular values, change in entropy and time!
    delta_nsv_arr = Float64[]
    delta_vn_ent_arr = Float64[]
    sf_vn_ent_arr = Float64[]
    time_arr = Float64[]
    linkdims_arr = Float64[]
    hsip_arr = Float64[]

    fid_val = F_HS(r_lpdo, r_lpdo_dc)
    push!(hsip_arr, fid_val)

    link_dims = linkdims(r_lpdo)
    link_dims_avg = sum(link_dims) / length(link_dims)
    push!(linkdims_arr, link_dims_avg)


    Random.seed!(r_seed)


    while layers < nlayers

        if linkdims(r_lpdo) == [1 for l in 1:(length(sites) - 1)]

            break

        else

            layers = layers + 1

            for odd_site in 1:2:(N - 1)

                if linkdim(r_lpdo, odd_site) != 1

                    orthogonalize!(r_lpdo, odd_site)

                    left_index = κ_ind_finder(r_lpdo[odd_site], odd_site)
                    right_index = κ_ind_finder(r_lpdo[odd_site + 1], odd_site + 1)

                    if odd_site == 1
                        left_neighbors = [sites[odd_site]]
                    else
                        left_neighbors = [linkind(r_lpdo, odd_site - 1), sites[odd_site]]
                    end

                    if odd_site != N - 1
                        right_neighbors = [linkind(r_lpdo, odd_site + 1), sites[odd_site + 1]]
                    else
                        right_neighbors = [sites[odd_site + 1]]
                    end

                    ## construct the nearest neighbor tensor
                    two_site_tensor = r_lpdo[odd_site] * r_lpdo[odd_site + 1]

                    svd_left_indices = vcat(left_neighbors, left_index)

                    # pre update SVD
                    _, _, s_arr_0, _ = chi_opt(r_lpdo, odd_site, odd_site + 1, opt_cutoff, false)

                    # perform the optimization on the nearest neighbors
                    opt_iso, _, _, _, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; kwargs...)

                    # apply the optimized isometry on the clubbed composite object
                    two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors, right_neighbors)

                    # post application of isometry
                    U, V, s_arr_1, _ = chi_opt(r_lpdo, odd_site, odd_site + 1, opt_cutoff, true; c_obj = two_site_tensor_update)

                    if (length(s_arr_0) == length(s_arr_1)) && all(isapprox(s_arr_0, s_arr_1))

                        opt_iso, _, _, _, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; initial_p0 = "RE", retraction_method = QRRetraction(), kwargs...)

                        # apply the optimized isometry on the clubbed composite object
                        two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors, right_neighbors)

                        # post update SVD
                        U, V, s_arr_1, _ = chi_opt(r_lpdo, odd_site, odd_site + 1, opt_cutoff, true; c_obj = two_site_tensor_update)

                        r_lpdo[odd_site] = U
                        r_lpdo[odd_site + 1] = V

                        push!(time_arr, elapsed_time)

                    else

                        r_lpdo[odd_site] = U
                        r_lpdo[odd_site + 1] = V

                        push!(time_arr, elapsed_time)

                    end

                    # record the difference in vn entropy before and after
                    vn_ent_pre = alpha_renyi_cutoff(s_arr_0, 1, 0.0)
                    vn_ent_post = alpha_renyi_cutoff(s_arr_1, 1, 0.0)

                    push!(sf_vn_ent_arr, vn_ent_post)

                    push!(delta_vn_ent_arr, vn_ent_pre - vn_ent_post)

                    # record the difference in singular values before and after
                    push!(delta_nsv_arr, length(s_arr_0) - length(s_arr_1))

                else


                end

            end


            for even_site in 2:2:(N - 2)

                if linkdim(r_lpdo, even_site) != 1

                    orthogonalize!(r_lpdo, even_site)

                    left_index = κ_ind_finder(r_lpdo[even_site], even_site)
                    right_index = κ_ind_finder(r_lpdo[even_site + 1], even_site + 1)

                    left_neighbors = [linkind(r_lpdo, even_site - 1), sites[even_site]]
                    right_neighbors = [linkind(r_lpdo, even_site + 1), sites[even_site + 1]]

                    ## construct the nearest neighbor tensor
                    two_site_tensor = r_lpdo[even_site] * r_lpdo[even_site + 1]

                    svd_left_indices = vcat(left_neighbors, left_index)

                    # pre update SVD
                    _, _, s_arr_0, _ = chi_opt(r_lpdo, even_site, even_site + 1, opt_cutoff, false) #; c_obj = mprho_ij)

                    # perform the optimization on the nearest neighbors
                    opt_iso, _, _, _, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; kwargs...)

                    # apply the optimized isometry on the clubbed composite object
                    two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors, right_neighbors)

                    # post application of isometry
                    U, V, s_arr_1, _ = chi_opt(r_lpdo, even_site, even_site + 1, opt_cutoff, true; c_obj = two_site_tensor_update)

                    if (length(s_arr_0) == length(s_arr_1)) && all(isapprox(s_arr_0, s_arr_1))

                        opt_iso, _, _, _, elapsed_time = opt_function(two_site_tensor, left_index, right_index, left_neighbors, right_neighbors; initial_p0 = "RE", retraction_method = QRRetraction(), kwargs...)

                        # apply the optimized isometry on the clubbed composite object
                        two_site_tensor_update = apply_optimized_isometry(opt_iso, two_site_tensor, left_index, right_index, left_neighbors, right_neighbors)

                        # post update SVD
                        U, V, s_arr_1, _ = chi_opt(r_lpdo, even_site, even_site + 1, opt_cutoff, true; c_obj = two_site_tensor_update)

                        r_lpdo[even_site] = U
                        r_lpdo[even_site + 1] = V

                        push!(time_arr, elapsed_time)

                    else

                        r_lpdo[even_site] = U
                        r_lpdo[even_site + 1] = V

                        push!(time_arr, elapsed_time)
                    end

                    # record the difference in vn entropy before and after
                    vn_ent_pre = alpha_renyi_cutoff(s_arr_0, 1, 0.0)
                    vn_ent_post = alpha_renyi_cutoff(s_arr_1, 1, 0.0)

                    push!(sf_vn_ent_arr, vn_ent_post)

                    push!(delta_vn_ent_arr, vn_ent_pre - vn_ent_post)

                    # record the difference in singular values before and after
                    push!(delta_nsv_arr, length(s_arr_0) - length(s_arr_1))

                else


                end

            end

        end

        link_dims = linkdims(r_lpdo)
        link_dims_avg = sum(link_dims) / length(link_dims)
        push!(linkdims_arr, link_dims_avg)

        fid_val = F_HS(r_lpdo, r_lpdo_dc)
        push!(hsip_arr, fid_val)


    end

    return layers, sf_vn_ent_arr, delta_vn_ent_arr, delta_nsv_arr, linkdims_arr, hsip_arr, time_arr

end
