"""
    alpha_renyi_cutoff(S, a, δ; renormalize = false, kwargs...)

Compute the ``\alpha``-Rényi entropy from a spectrum of singular values `S` using a cutoff
threshold `δ` to filter out negligible contributions.

## Arguments
- `S`: Spectrum of singular values (can be an array, vector, or ITensor spectrum object).
- `a`: Power parameter ``α`` for the Rényi entropy (supports ``α`` = 1 for von Neumann entropy).
- `δ`: Cutoff threshold below which singular values are ignored.

## Keywords
- `renormalize = false`: Boolean flag to normalize the singular value spectrum before entropy calculation.
- `kwargs...`: Additional keyword arguments (reserved for compatibility).

## Returns
- The computed ``α``-Rényi entropy value.
"""
function alpha_renyi_cutoff(S, a, δ; renormalize = false, kwargs...)

    ent = 0

    sv = Array(S)

    if renormalize == true
        # println("renormalization on!")
        sv = normalize(sv)
    end

    if a == 1

        for s2 in sv
            if s2 > δ
                ent = ent - (s2^2) * log(s2^2)
            end
        end

        return ent

    else

        for s2 in sv
            if s2 > δ
                ent = ent + (s2)^(2 * a)
            end

        end

        ent = (1 / (1 - a)) * log(ent)

        return ent
    end


end

# function entropy_cutoff(S, δ)

#     ent = 0
#     p = 0
    # sq_sum = 0
    # sv = Array(diag(S))
#     sv = Array(S)
#     de = sum([s1^2 for s1 in sv])

#     for s2 in sv
#         if s2 > δ
            # ent = ent - (s2^2/de)*log(s2^2/de)
#             ent = ent - (s2^2) * log(s2^2)
            # sq_sum  = sq_sum + s2^2
#         end
#     end

#     return ent #+ 0.1*p) # + 0.5*(sq_sum - 1)

# end

"""
    apply_optimized_isometry(opt_isometry, composite_object, left_index, right_index,
                             left_neighbors, right_neighbors)

Apply an optimized isometry tensor to a composite tensor object, verify its unitarity,
and return the updated tensor split to match the pre-application setting.

## Arguments
- `opt_isometry`: The optimized isometry as an `ITensor` with primed and unprimed combined indices.
- `composite_object`: Composite `ITensor` object on which the isometry is applied.
- `left_index`: Left index out of the two indices on which the isometry acts.
- `right_index`: Right index out of the two indices on which the isometry acts.
- `left_neighbors`: Array of indices representing the left environment.
- `right_neighbors`: Array of indices representing the right environment.

## Returns
- The updated composite `ITensor` after isometry application with the same data structure as pre-application.

See also: [`isometry_ro_are`](@ref), [`isometry_ro_sre`](@ref), [`isometry_ts`](@ref), [`isometry_ev`](@ref)
"""
function apply_optimized_isometry(opt_isometry, composite_object, left_index, right_index, left_neighbors, right_neighbors)

    # optimal isometry as combined index so generate that again in the composite object
    combined_tensor = combiner(left_index, right_index; tags = "comb")
    combined_index = combinedind(combined_tensor)
    composite_tensor = composite_object * combined_tensor

    # replace the above combined index to match the index of optimal isometry
    composite_tensor = composite_tensor * delta(combined_index, noprime(inds(opt_isometry)[1]))

    # check if isometry is actually an isometry!
    # opt_iso_arr = Array(opt_isometry.tensor)
    # @assert isapprox(opt_iso_arr * adjoint(opt_iso_arr), Matrix((1.0 + 0.0im)I, size(opt_iso_arr))) "Isometry check fail!"
    # @assert isapprox(adjoint(opt_iso_arr) * opt_iso_arr, Matrix((1.0 + 0.0im)I, size(opt_iso_arr))) "Isometry check fail!"

    # apply the isometry
    composite_tensor = opt_isometry * composite_tensor
    composite_tensor = replaceind(composite_tensor, noprime(inds(opt_isometry)[1])' => noprime(inds(opt_isometry)[1]))

    # permute to get the combined_index to the beginning!
    # ordering is crucial to compute the entropy at the right cut!
    ordering_indices = [noprime(inds(opt_isometry)[1])]
    if length(left_neighbors) != 0
        append!(ordering_indices, left_neighbors)
    end
    if length(right_neighbors) != 0
        append!(ordering_indices, right_neighbors)
    end

    # permute with combined index at the start
    composite_tensor = permute(composite_tensor, ordering_indices)

    # split the composite tensor, TODO: Gauge freedom in how we split the indices
    composite_tensor_mat = Array(composite_tensor.tensor)

    # split the combined index into left index and right index
    if length(left_neighbors) == 0
        composite_tensor_mat = reshape(composite_tensor_mat, dim(left_index), dim(right_index), prod(dim.(right_neighbors)))
    elseif length(right_neighbors) == 0
        composite_tensor_mat = reshape(composite_tensor_mat, dim(left_index), dim(right_index), prod(dim.(left_neighbors)))
    else
        composite_tensor_mat = reshape(composite_tensor_mat, dim(left_index), dim(right_index), prod(dim.(left_neighbors)) * prod(dim.(right_neighbors)))
    end

    # permute the indices neighbor and the right index
    composite_tensor_mat = permutedims(composite_tensor_mat, (1, 3, 2))

    # generate a new index array with the permuted order
    split_ordering = [left_index]
    if length(left_neighbors) != 0
        append!(split_ordering, left_neighbors)
    end
    if length(right_neighbors) != 0
        append!(split_ordering, right_neighbors)
    end
    push!(split_ordering, right_index)

    split_ordering_dims = dim.(split_ordering)

    # reshape the tensor mat with the above dimensions
    composite_tensor_mat = reshape(composite_tensor_mat, Tuple(split_ordering_dims))

    # create the new composite_tensor
    return composite_tensor_new = ITensor(composite_tensor_mat, split_ordering)

end

"""
    permute_state_and_return_raw_data(psi, set_dag)

Permute the site and link indices of a Matrix Product State (MPS) `psi` into a standardized
canonical order across all tensors and extract their raw multi-dimensional array data.

## Arguments
- `psi`: The Matrix Product State (`MPS`) object from `ITensors.jl`.
- `set_dag`: Boolean flag; if `true`, takes the conjugate/dagger (`dag`) of the state before processing.

## Returns
- A `Vector{Array}` containing the raw tensor array data for each site in the MPS, properly permuted
  along their link and site indices.

See also: [`inner_custom`](@ref)
"""
function permute_state_and_return_raw_data(psi, set_dag)

    psi_dc = deepcopy(psi)

    if set_dag == true

        psi_dc = dag(psi_dc)

    end

    sites = siteinds(psi_dc)

    raw_tensor_data = Vector{Array}(undef, length(psi_dc))

    psi_dc[1] = permute(psi_dc[1], (sites[1], linkinds(psi_dc)[1]))

    raw_tensor_data[1] = Array(psi_dc[1].tensor)

    for i in 2:(length(psi_dc) - 1)

        psi_dc[i] = permute(psi_dc[i], (linkinds(psi_dc)[i - 1], sites[i], linkinds(psi_dc)[i]))

        raw_tensor_data[i] = Array(psi_dc[i].tensor)

    end

    psi_dc[end] = permute(psi_dc[end], (linkinds(psi_dc)[end], sites[end]))

    raw_tensor_data[end] = Array(psi_dc[end].tensor)

    return raw_tensor_data

end

# eta is daggered!
function inner_custom_full_wfs(psi_p_arr, eta_dag_p_arr)

    # psi_p_arr = permute_state_and_return_raw_data(psi, false)
    # eta_dag_p_arr = permute_state_and_return_raw_data(eta, true)

    # the structure of each tensor is left_link = site = right_link
    prod = 1

    # put the sites together and reshape
    psi1 = psi_p_arr[1]
    psi1_p = permutedims(psi1, (2, 1))
    eta1 = eta_dag_p_arr[1]
    eta1_p = permutedims(eta1, (1, 2))

    prod = prod * psi1_p * eta1_p
    prod = reshape(prod, (1, size(psi1)[2] * size(eta1)[2]))

    N = length(psi_p_arr)

    # run through the chain
    for c in 2:(N - 1)

        prodc = 1

        # put the sites together, structure of each tensor: left_link = site = right_link
        psic = psi_p_arr[c]
        psic_p = permutedims(psic, (1, 3, 2))
        psic_pr = reshape(psic_p, size(psic)[1] * size(psic)[3], size(psic)[2])

        etac = eta_dag_p_arr[c]
        etac_p = permutedims(etac, (2, 1, 3))
        etac_pr = reshape(etac_p, size(etac)[2], size(etac)[1] * size(etac)[3])

        prodc = prodc * psic_pr * etac_pr

        prodc = reshape(prodc, size(psic)[1], size(psic)[3], size(etac)[1], size(etac)[3])
        prodc = permutedims(prodc, (1, 3, 2, 4))

        prodc = reshape(prodc, size(psic)[1] * size(etac)[1], size(psic)[3] * size(etac)[3])

        prod = prod * prodc

    end

    psiN = psi_p_arr[N]
    psiN_p = permutedims(psiN, (1, 2))
    etaN = eta_dag_p_arr[N]
    etaN_p = permutedims(etaN, (2, 1))

    prodc = psiN_p * etaN_p
    prodc = reshape(prodc, (size(psiN)[1] * size(etaN)[1], 1))

    prod = prod * prodc

    println("here: ", typeof(prod[1]), " ", prod[1])

    return prod[1]

end

function inner_custom_bulk(two_site_tensor, i, j, psi_p_arr, eta_dag_p_arr)

    # both the arrays are left_link = site = right_link
    # two_site_tensor comes as left_link = two sites = right link

    prod = 1

    # put the sites together and reshape
    psi1 = psi_p_arr[1]
    psi1_p = permutedims(psi1, (2, 1))
    eta1 = eta_dag_p_arr[1]
    eta1_p = permutedims(eta1, (1, 2))

    prod = prod * psi1_p * eta1_p
    prod = reshape(prod, (1, size(psi1)[2] * size(eta1)[2]))

    N = length(psi_p_arr)

    # run through the chain
    for c in 2:(N - 1)

        prodc = 1

        if c == i

            ts_tensor_r = reshape(two_site_tensor, size(two_site_tensor)[1], size(two_site_tensor)[2] * size(two_site_tensor)[3], size(two_site_tensor)[4])
            ts_tensor_rp = permutedims(ts_tensor_r, (1, 3, 2))
            ts_tensor_rpr = reshape(ts_tensor_rp, size(ts_tensor_r)[1] * size(ts_tensor_r)[3], size(ts_tensor_r)[2])

            etac = eta_dag_p_arr[c]
            etac_r = reshape(etac, size(etac)[1] * size(etac)[2], size(etac)[3])
            etacp1 = eta_dag_p_arr[c + 1]
            etacp1_r = reshape(etacp1, size(etacp1)[1], size(etacp1)[2] * size(etacp1)[3])

            etac_tensor = etac_r * etacp1_r
            etac_tensor_r = reshape(etac_tensor, size(etac)[1], size(etac)[2] * size(etacp1)[2], size(etacp1)[3])
            etac_tensor_rp = permutedims(etac_tensor_r, (2, 1, 3))
            etac_tensor_rpr = reshape(etac_tensor_rp, size(etac_tensor_r)[2], size(etac_tensor_r)[1] * size(etac_tensor_r)[3])

            prodc = prodc * ts_tensor_rpr * etac_tensor_rpr

            prodc = reshape(prodc, size(ts_tensor_r)[1], size(ts_tensor_r)[3], size(etac_tensor_r)[1], size(etac_tensor_r)[3])
            prodc = permutedims(prodc, (1, 3, 2, 4))

            prodc = reshape(prodc, size(ts_tensor_r)[1] * size(etac_tensor_r)[1], size(ts_tensor_r)[3] * size(etac_tensor_r)[3])

            prod = prod * prodc


        elseif c == i + 1

        else

            # put the sites together, structure of each tensor: left_link = site = right_link
            psic = psi_p_arr[c]
            psic_p = permutedims(psic, (1, 3, 2))
            psic_pr = reshape(psic_p, size(psic)[1] * size(psic)[3], size(psic)[2])

            etac = eta_dag_p_arr[c]
            etac_p = permutedims(etac, (2, 1, 3))
            etac_pr = reshape(etac_p, size(etac)[2], size(etac)[1] * size(etac)[3])

            prodc = prodc * psic_pr * etac_pr

            prodc = reshape(prodc, size(psic)[1], size(psic)[3], size(etac)[1], size(etac)[3])
            prodc = permutedims(prodc, (1, 3, 2, 4))

            prodc = reshape(prodc, size(psic)[1] * size(etac)[1], size(psic)[3] * size(etac)[3])

            prod = prod * prodc

        end


    end

    psiN = psi_p_arr[N]
    psiN_p = permutedims(psiN, (1, 2))
    etaN = eta_dag_p_arr[N]
    etaN_p = permutedims(etaN, (2, 1))

    prodc = psiN_p * etaN_p
    prodc = reshape(prodc, (size(psiN)[1] * size(etaN)[1], 1))

    prod = prod * prodc

    return prod[1]

end

function inner_custom_left_boundary(two_site_tensor, psi_p_arr, eta_dag_p_arr)

    prod = 1

    two_site_tensor_r = reshape(two_site_tensor, size(two_site_tensor)[1] * size(two_site_tensor)[2], size(two_site_tensor)[3])
    two_site_tensor_rp = permutedims(two_site_tensor_r, (2, 1))

    eta1 = eta_dag_p_arr[1]
    eta1_r = reshape(eta1, size(eta1)[1], size(eta1)[2])
    eta2 = eta_dag_p_arr[2]
    eta2_r = reshape(eta2, size(eta2)[1], size(eta2)[2] * size(eta2)[3])
    eta12 = eta1_r * eta2_r

    eta_12_r = reshape(eta12, size(eta1)[1] * size(eta2)[2], size(eta2)[3])

    prod = prod * two_site_tensor_rp * eta_12_r
    prod = reshape(prod, 1, (size(two_site_tensor)[3] * size(eta2)[3]))

    N = length(psi_p_arr)

    for c in 3:(N - 1)

        prodc = 1

        # put the sites together, structure of each tensor: left_link = site = right_link
        psic = psi_p_arr[c]
        psic_p = permutedims(psic, (1, 3, 2))
        psic_pr = reshape(psic_p, size(psic)[1] * size(psic)[3], size(psic)[2])

        etac = eta_dag_p_arr[c]
        etac_p = permutedims(etac, (2, 1, 3))
        etac_pr = reshape(etac_p, size(etac)[2], size(etac)[1] * size(etac)[3])

        prodc = prodc * psic_pr * etac_pr

        prodc = reshape(prodc, size(psic)[1], size(psic)[3], size(etac)[1], size(etac)[3])
        prodc = permutedims(prodc, (1, 3, 2, 4))

        prodc = reshape(prodc, size(psic)[1] * size(etac)[1], size(psic)[3] * size(etac)[3])

        prod = prod * prodc

    end

    psiN = psi_p_arr[N]
    psiN_p = permutedims(psiN, (1, 2))
    etaN = eta_dag_p_arr[N]
    etaN_p = permutedims(etaN, (2, 1))

    prodc = psiN_p * etaN_p
    prodc = reshape(prodc, (size(psiN)[1] * size(etaN)[1], 1))

    prod = prod * prodc

    return prod[1]

end

function inner_custom_right_boundary(two_site_tensor, psi_p_arr, eta_dag_p_arr)

    # both the arrays are left_link = site = right_link
    # two_site_tensor comes as left_link = two sites = right link

    prod = 1

    # put the sites together and reshape
    psi1 = psi_p_arr[1]
    psi1_p = permutedims(psi1, (2, 1))
    eta1 = eta_dag_p_arr[1]
    eta1_p = permutedims(eta1, (1, 2))

    prod = prod * psi1_p * eta1_p
    prod = reshape(prod, (1, size(psi1)[2] * size(eta1)[2]))

    N = length(psi_p_arr)

    for c in 2:(N - 2)

        prodc = 1

        # put the sites together, structure of each tensor: left_link = site = right_link
        psic = psi_p_arr[c]
        psic_p = permutedims(psic, (1, 3, 2))
        psic_pr = reshape(psic_p, size(psic)[1] * size(psic)[3], size(psic)[2])

        etac = eta_dag_p_arr[c]
        etac_p = permutedims(etac, (2, 1, 3))
        etac_pr = reshape(etac_p, size(etac)[2], size(etac)[1] * size(etac)[3])

        prodc = prodc * psic_pr * etac_pr

        prodc = reshape(prodc, size(psic)[1], size(psic)[3], size(etac)[1], size(etac)[3])
        prodc = permutedims(prodc, (1, 3, 2, 4))

        prodc = reshape(prodc, size(psic)[1] * size(etac)[1], size(psic)[3] * size(etac)[3])

        prod = prod * prodc

    end

    two_site_tensor_r = reshape(two_site_tensor, size(two_site_tensor)[1], size(two_site_tensor)[2] * size(two_site_tensor)[3])

    eta1 = eta_dag_p_arr[N - 1]
    eta1_r = reshape(eta1, size(eta1)[1] * size(eta1)[2], size(eta1)[3])
    eta2 = eta_dag_p_arr[N]
    eta2_r = reshape(eta2, size(eta2)[1], size(eta2)[2])

    eta_12 = eta1_r * eta2_r

    eta_12_r = reshape(eta_12, size(eta1)[1], size(eta1)[2] * size(eta2)[2])
    eta_12_rp = permutedims(eta_12_r, (2, 1))

    prodc = two_site_tensor_r * eta_12_rp
    prodc = reshape(prodc, size(two_site_tensor)[1] * size(eta1)[1], 1)

    prod = prod * prodc

    return prod[1]


end

"""
    inner_custom(two_site_tensor, i, psi_p_arr, eta_dag_p_arr)

Dispatch and compute a custom inner product between `psi_p_arr` and `eta_dag_p_arr` while taking
into consideration the already contracted two-site tensor at position `i` and `i+1` in `psi_p_arr`.

## Arguments
- `two_site_tensor`: The local two-site tensor object
- `i`: Integer site index indicating the position of the two-site tensor.
- `psi_p_arr`: Vector of raw tensor data for the primary state.
- `eta_dag_p_arr`: Vector of raw tensor data for the conjugate environment or state.

## Returns
- The result of the custom inner product contraction for the specified boundary or bulk region.
"""
function inner_custom(two_site_tensor, i, psi_p_arr, eta_dag_p_arr)

    N = length(psi_p_arr)

    if i == 1

        return inner_custom_left_boundary(two_site_tensor, psi_p_arr, eta_dag_p_arr)

    elseif i == N - 1

        return inner_custom_right_boundary(two_site_tensor, psi_p_arr, eta_dag_p_arr)

    else

        return inner_custom_bulk(two_site_tensor, i, i + 1, psi_p_arr, eta_dag_p_arr)

    end

end


"""
    permute_op_and_return_raw_data(op, set_dag)

Permutes the indices of a Matrix Product Operator (MPO) or similar tensor network
operator into a standard canonical site-index ordering and extracts its raw underlying
tensor data as a vector of standard Arrays.

### Arguments
- `op`: An MPO or tensor network operator (typically from ITensors.jl).
- `set_dag`: A boolean (`Bool`) flag. If `true`, the adjoint (dagger/conjugate transpose)
  of the operator is taken via `dag(op)` before processing.

### Returns
- `Vector{Array}`: A vector containing the raw dense array data extracted from each
  tensor in the permuted operator chain.
"""
function permute_op_and_return_raw_data(op, set_dag)

    N = length(op)

    op_dc = deepcopy(op)

    if set_dag == true

        op_dc = dag(op_dc)

    end

    sites = [noprime(uniqueind(siteinds(op_dc)[_k])) for _k in 1:N]

    raw_tensor_data = Vector{Array}(undef, length(op_dc))

    op_dc[1] = permute(op_dc[1], (sites[1], sites[1]', linkinds(op_dc)[1]))

    raw_tensor_data[1] = Array(op_dc[1].tensor)

    for i in 2:(length(op_dc) - 1)

        op_dc[i] = permute(op_dc[i], (linkinds(op_dc)[i - 1], sites[i], sites[i]', linkinds(op_dc)[i]))

        raw_tensor_data[i] = Array(op_dc[i].tensor)

    end

    op_dc[end] = permute(op_dc[end], (linkinds(op_dc)[end], sites[end], sites[end]'))

    raw_tensor_data[end] = Array(op_dc[end].tensor)

    return raw_tensor_data

end

"""
    normalized_Frobenius_norm_opt(two_site_arr_dc, raw_ref_op_dag, i)

Compute a normalized Frobenius norm or optimized overlap contraction for a two-site tensor
acting on neighboring sites `i` and `i + 1` against a reordered raw reference operator chain.

## Arguments
- `two_site_arr_dc`: Raw array data of the two-site tensor (deep-copied internally for safe manipulation).
- `raw_ref_op_dag`: Vector of reordered raw tensors representing the reference operator's adjoint chain.
- `i`: Integer site index indicating the starting position of the two-site operator.

## Returns
- The scalar contraction result representing the normalized norm/overlap value.
"""
function normalized_Frobenius_norm_opt(two_site_arr_dc, raw_ref_op_dag, i)

    two_site_arr = deepcopy(two_site_arr_dc)

    N = length(raw_ref_op_dag)

    # second tensor, second leg
    site_dim = size(raw_ref_op_dag[2])[2]

    if i == 1

        prod = 1

        # two_site_arr = Array(two_site_tensor.tensor)

        two_site_arr = reshape(two_site_arr, size(two_site_arr)[1] * size(two_site_arr)[2], size(two_site_arr)[3])

        id_op_ts = Matrix(1.0I, site_dim^2, site_dim^2)

        id_op_ts_re = reshape(id_op_ts, 1, site_dim^4)

        prod = id_op_ts_re * two_site_arr

        ###
        id_op = Matrix(1.0I, site_dim, site_dim)

        id_op_re = reshape(id_op, 1, site_dim * site_dim)

        for c in 3:(N - 1)

            ten_c = raw_ref_op_dag[c]

            ten_cr = reshape(ten_c, size(ten_c)[1], size(ten_c)[2] * size(ten_c)[3] * size(ten_c)[4])

            prod = prod * ten_cr

            prod = reshape(prod, size(ten_c)[2] * size(ten_c)[3], size(ten_c)[4])

            prod = id_op_re * prod

        end

        last_ten = raw_ref_op_dag[N]

        last_ten_r = reshape(last_ten, size(last_ten)[1], size(last_ten)[2] * size(last_ten)[3])

        prod = prod * last_ten_r

        prod = reshape(prod, size(last_ten)[2] * size(last_ten)[3], 1)

        prod = id_op_re * prod

        return prod[1, 1]


    elseif i == N - 1

        prod = 1

        id_op = Matrix(1.0I, site_dim, site_dim)

        id_op_re = reshape(id_op, 1, site_dim * site_dim)

        first_ten = raw_ref_op_dag[1]

        first_ten = reshape(first_ten, size(first_ten)[1] * size(first_ten)[2], size(first_ten)[3])

        prod = prod * id_op_re * first_ten

        for c in 2:(N - 2)

            ten_c = raw_ref_op_dag[c]
            ten_cr = reshape(ten_c, size(ten_c)[1], size(ten_c)[2] * size(ten_c)[3] * size(ten_c)[4])

            prod = prod * ten_cr

            prod = reshape(prod, size(ten_c)[2] * size(ten_c)[3], size(ten_c)[4])

            prod = id_op_re * prod

        end

        # two_site_arr = Array(two_site_tensor.tensor)

        two_site_arr_r = reshape(two_site_arr, size(two_site_arr)[1], size(two_site_arr)[2] * size(two_site_arr)[3])

        prod = prod * two_site_arr_r

        prod = reshape(prod, size(two_site_arr)[2] * size(two_site_arr)[3], 1)

        id_op_ts = Matrix(1.0I, site_dim^2, site_dim^2)

        id_op_ts_re = reshape(id_op_ts, 1, site_dim^4)

        prod = id_op_ts_re * prod

        return prod[1, 1]


    else

        # go from 1 to i-1 and then contract two_site_tensor and the rest of the chain i+2 to N

        prod = 1

        id_op = Matrix(1.0I, site_dim, site_dim)

        id_op_re = reshape(id_op, 1, site_dim * site_dim)

        first_ten = raw_ref_op_dag[1]

        first_ten = reshape(first_ten, size(first_ten)[1] * size(first_ten)[2], size(first_ten)[3])

        prod = prod * id_op_re * first_ten

        for c in 2:(i - 1)

            ten_c = raw_ref_op_dag[c]
            ten_cr = reshape(ten_c, size(ten_c)[1], size(ten_c)[2] * size(ten_c)[3] * size(ten_c)[4])

            prod = prod * ten_cr

            prod = reshape(prod, size(ten_c)[2] * size(ten_c)[3], size(ten_c)[4])

            prod = id_op_re * prod

        end

        # two_site_arr = Array(two_site_tensor.tensor)

        two_site_arr_r = reshape(two_site_arr, size(two_site_arr)[1], size(two_site_arr)[2] * size(two_site_arr)[3] * size(two_site_arr)[4])

        prod = prod * two_site_arr_r

        prod = reshape(prod, size(two_site_arr)[2] * size(two_site_arr)[3], size(two_site_arr)[4])

        id_op_ts = Matrix(1.0I, site_dim^2, site_dim^2)

        id_op_ts_re = reshape(id_op_ts, 1, site_dim^4)

        prod = id_op_ts_re * prod

        for c in (i + 2):(N - 1)

            ten_c = raw_ref_op_dag[c]
            ten_cr = reshape(ten_c, size(ten_c)[1], size(ten_c)[2] * size(ten_c)[3] * size(ten_c)[4])

            prod = prod * ten_cr

            prod = reshape(prod, size(ten_c)[2] * size(ten_c)[3], size(ten_c)[4])

            prod = id_op_re * prod

        end

        last_ten = raw_ref_op_dag[N]

        last_ten_r = reshape(last_ten, size(last_ten)[1], size(last_ten)[2] * size(last_ten)[3])

        prod = prod * last_ten_r

        prod = reshape(prod, size(last_ten)[2] * size(last_ten)[3], 1)

        prod = id_op_re * prod

        return prod[1, 1]

    end

end

export entropy_cutoff, alpha_renyi_cutoff, apply_optimized_isometry, permute_state_and_return_raw_data, inner_custom_full_wfs, inner_custom, permute_op_and_return_raw_data, normalized_Frobenius_norm_opt
