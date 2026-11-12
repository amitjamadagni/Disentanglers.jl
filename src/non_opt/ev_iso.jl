"""
    isometry_ev_it(composite_object, left_index, right_index, left_neighbors, right_neighbors;
                   manifold = Stiefel, manifold_field = ℂ, initial_p0 = "RS", p0_val = nothing, kwargs...)

Construct and evaluate an optimized isometry that disentangles a composite tensor network
object using environment contraction methods. (Referred to as `Evenbly-Vidal` construction).
The method relies on `ITensor` data types and tensor contraction routines from `ITensors.jl`.

!!! note
    Because this routine relies on `ITensors.jl` contraction mechanics, it may be slower than raw multi-dimesional array alternatives
    due to metadata handling.

## Arguments
- `composite_object`: Composite `ITensor` object on which the isometry acts.
- `left_index`: Left index out of the two indices on which the isometry acts.
- `right_index`: Right index out of the two indices on which the isometry acts.
- `left_neighbors`: Array of indices representing the left environment.
- `right_neighbors`: Array of indices representing the right environment.

## Keywords
- `manifold = Stiefel`: Manifold constructor for optimization (e.g., `Stiefel`, can also be `Unitary`, `Grassmannian`).
- `manifold_field = ℂ`: Number field type over which the manifold is defined (defaults to Complex, ℂ but can also be Real, ℝ field).
- `initial_p0 = "RS"`: Strategy for initial point generation (`"RS"`: `Random Split`, `"RE"`: `Random Entangled`, `"I"`: `Identity`, or `"C"`: `Custom`).
- `p0_val = nothing`: Custom initial matrix value if `initial_p0 = "C"`.
- `kwargs...`: Additional keyword arguments (reserved for compatibility).

## Returns
- `error_loop`: A boolean flag indicating error status (`false` upon successful execution). #TODO
- `u_kij`: Computed isometry as an `ITensor` with primed and unprimed combined indices.
- `elapsed_time`: Total execution time in seconds.

See also: [`isometry_ro_are`](@ref), [`isometry_ro_sre`](@ref), [`isometry_ts`](@ref), [`isometry_ev`](@ref)
"""
function isometry_ev_it(composite_object, left_index, right_index, left_neighbors, right_neighbors; manifold = Stiefel, manifold_field = ℂ, initial_p0 = "RS", p0_val = nothing, kwargs...)

    # always check for the orthogonality center before applying the isometry!

    # generate a random isometry that acts on the clubbed space
    if initial_p0 == "RS"
        M1 = manifold(dim(left_index), dim(left_index), manifold_field)
        M2 = manifold(dim(right_index), dim(right_index), manifold_field)
        p0 = kron(rand(M1), rand(M2))
    elseif initial_p0 == "RE"
        M = manifold(dim(left_index) * dim(right_index), dim(left_index) * dim(right_index), manifold_field)
        p0 = rand(M)
    elseif initial_p0 == "I"
        # id_m = Matrix((1.0+0.0im)I, dim(left_index), dim(right_index))
        # p0 = kron(id_m, id_m)
        p0 = Matrix((1.0 + 0.0im)I, dim(left_index) * dim(right_index), dim(left_index) * dim(right_index))
    elseif initial_p0 == "C" && p0_val != nothing
        p0 = p0_val
    end

    # this is a fix for MPO!
    rand_iso_it = ITensor(p0, left_index'', right_index'', left_index, right_index)

    start_time = time()

    # we can think of the algorithm as four different blocks
    # block 1
    composite_object_b1 = rand_iso_it * composite_object
    composite_object_b1 = mapprime(composite_object_b1, 2, 0)

    # block 2
    composite_object_b2 = deepcopy(composite_object_b1)
    composite_object_b2 = dag(composite_object_b2)'

    # block 3
    composite_object_b3 = deepcopy(composite_object)
    composite_object_b3 = (composite_object)''

    # block 4
    composite_object_b4 = deepcopy(composite_object_b1)
    composite_object_b4 = dag(composite_object_b4)'''

    ### implement the contraction as in arxiv: 2509.16439
    ## two strategies either we create a combiner of all the right neighbors and delta
    ## 								or
    ## run a loop through all the neighbors while delta'ing
    ## we do the second of the above options!

    ## put the right neigbhors from block 1 and block 2 including the right index!
    composite_object_b12 = composite_object_b1 * composite_object_b2
    for rn in right_neighbors
        composite_object_b12 = composite_object_b12 * delta(rn, rn')
    end
    composite_object_b12 = composite_object_b12 * delta(right_index, right_index')

    # put the right neigbhors from block 3 and block 4 excluding the right index!
    composite_object_b34 = composite_object_b3 * composite_object_b4
    for rn in right_neighbors
        composite_object_b34 = composite_object_b34 * delta(rn'', rn''')
    end

    # put 12 and 34 together to create the env tensor

    # contract the left neighbors!
    env_tensor = composite_object_b12 * composite_object_b34
    for ln in left_neighbors
        env_tensor = env_tensor * delta(ln, ln''')
        env_tensor = env_tensor * delta(ln', ln'')
    end
    env_tensor = env_tensor * delta(left_index''', left_index)

    env_tensor = permute(env_tensor, left_index', right_index''', left_index'', right_index'')

    env_arr = Array(env_tensor.tensor)
    env_arr = reshape(env_arr, dim(left_index) * dim(right_index), dim(left_index) * dim(right_index))

    # perform the SVD and extract the isometry!
    U, S, V = svd(env_tensor, (left_index', right_index'''); cutoff = 0.0)

    U_p = permute(U, left_index', right_index''', commoninds(U, S)[1])
    V_p = permute(V, commoninds(S, V)[1], left_index'', right_index'')

    U_p_mat = Array(U_p.tensor)
    U_p_mat_re = reshape(U_p_mat, dim(left_index) * dim(right_index), dim(commoninds(U, S)[1]))

    V_p_mat = Array(V_p.tensor)
    V_p_mat_re = reshape(V_p_mat, dim(commoninds(S, V)[1]), dim(left_index) * dim(right_index))

    u_kij_arr = adjoint(V_p_mat_re) * adjoint(U_p_mat_re)

    elapsed_time = time() - start_time

    combined_tensor = combiner(left_index, right_index; tags = "comb")
    combined_index = combinedind(combined_tensor)

    # here which index to prime is a bit of ambiguity!
    # the following convention reduces the entropy ...
    u_kij = ITensor(u_kij_arr, combined_index', combined_index)

    return false, u_kij, elapsed_time


end

"""
    isometry_ev(composite_object, left_index, right_index, left_neighbors, right_neighbors;
                ev_original = false, manifold = Stiefel, manifold_field = ℂ,
                initial_p0 = "RS", p0_val = nothing, kwargs...)

Construct and evaluate an optimized isometry that disentangles a composite tensor network
object using environment contraction methods. (Referred to as `Evenbly-Vidal` construction).

!!! note
    Because this routine directly manipulates multi-dimensional arrays rather than leveraging `ITensors.jl` internal data structures, it is heavily optimized for performance and lower overhead.

!!! note
    This function is designed to be used in conjunction with [`apply_optimized_isometry`](@ref) in order
    to return to the original form. See examples (add reference to the example).

## Arguments
- `composite_object`: Composite `ITensor` object on which the isometry acts.
- `left_index`: Left index out of the two indices on which the isometry acts.
- `right_index`: Right index out of the two indices on which the isometry acts.
- `left_neighbors`: Array of indices representing the left environment.
- `right_neighbors`: Array of indices representing the right environment.

## Keywords
- `ev_original = false`: Boolean flag to toggle between alternative isometry contraction orderings (`F.V * adjoint(F.U)` (original) vs `adjoint(F.U) * F.V` (optimized)).
- `manifold = Stiefel`: Manifold constructor for optimization (e.g., `Stiefel`, can also be `Unitary`, `Grassmannian`).
- `manifold_field = ℂ`: Number field type over which the manifold is defined (defaults to Complex, ℂ but can also be Real, ℝ field).
- `initial_p0 = "RS"`: Strategy for initial point generation (`"RS"`: `Random Split`, `"RE"`: `Random Entangled`, `"I"`: `Identity`, or `"C"`: `Custom`).
- `p0_val = nothing`: Custom initial matrix value if `initial_p0 = "C"`.
- `kwargs...`: Additional keyword arguments (reserved for compatibility).

## Returns
- `error_loop`: A boolean flag indicating error status (`false` upon successful execution).
- `u_kij`: Computed isometry as an `ITensor` with primed and unprimed combined indices.
- `elapsed_time`: Total execution time in seconds.

## References
The environment contraction scheme and raw array optimization techniques implemented here are
based on the construction described in:
- arXiv:2509.16439.

See also: [`isometry_ro_are`](@ref), [`isometry_ro_sre`](@ref), [`isometry_ts`](@ref),  [`isometry_ev_it`](@ref)
"""
function isometry_ev(composite_object, left_index, right_index, left_neighbors, right_neighbors; ev_original = false, manifold = Stiefel, manifold_field = ℂ, initial_p0 = "RS", p0_val = nothing, kwargs...)

    # always check for the orthogonality center before applying the isometry!

    # generate a random isometry that acts on the clubbed space
    if initial_p0 == "RS"
        M1 = manifold(dim(left_index), dim(left_index), manifold_field)
        M2 = manifold(dim(right_index), dim(right_index), manifold_field)
        p0 = kron(rand(M1), rand(M2))
    elseif initial_p0 == "RE"
        M = manifold(dim(left_index) * dim(right_index), dim(left_index) * dim(right_index), manifold_field)
        p0 = rand(M)
    elseif initial_p0 == "I"
        p0 = Matrix((1.0 + 0.0im)I, dim(left_index) * dim(right_index), dim(left_index) * dim(right_index))
    elseif initial_p0 == "C" && p0_val != nothing
        p0 = p0_val
    end

    combined_ln_tensor = nothing
    combined_rn_tensor = nothing

    if length(left_neighbors) != 0
        combined_ln_tensor = combiner(left_neighbors; tags = "ln_comb")
        combined_ln_index = combinedind(combined_ln_tensor)
    end

    # combine all the right neighbors into a single index
    if length(right_neighbors) != 0
        combined_rn_tensor = combiner(right_neighbors; tags = "rn_comb")
        combined_rn_index = combinedind(combined_rn_tensor)
    end

    composite_tensor = deepcopy(composite_object)

    if combined_ln_tensor != nothing
        composite_tensor = composite_tensor * combined_ln_tensor
    end
    if combined_rn_tensor != nothing
        composite_tensor = composite_tensor * combined_rn_tensor
    end


    dim_left_index = dim(left_index)
    dim_right_index = dim(right_index)

    if combined_ln_tensor != nothing && combined_rn_tensor != nothing
        composite_tensor = permute(composite_tensor, left_index, right_index, combined_ln_index, combined_rn_index)
        composite_tensor_arr = Array(composite_tensor.tensor)

    elseif combined_ln_tensor != nothing
        composite_tensor = permute(composite_tensor, left_index, right_index, combined_ln_index)
        composite_tensor_arr = Array(composite_tensor.tensor)
        composite_tensor_arr = reshape(composite_tensor_arr, dim(left_index), dim(right_index), dim(combined_ln_index), 1)

    elseif combined_rn_tensor != nothing
        composite_tensor = permute(composite_tensor, left_index, right_index, combined_rn_index)
        composite_tensor_arr = Array(composite_tensor.tensor)
        composite_tensor_arr = reshape(composite_tensor_arr, dim(left_index), dim(right_index), 1, dim(combined_rn_index))
    end

    #####################

    start_time = time()


    # composite_tensor_arr has ordering: left_index, right_index, prod(left_neighbor_arr), prod(right_neighbor_arr)
    size_cta = size(composite_tensor_arr)

    # we can think of the algorithm as four different blocks
    # block 1 ordering: li × ri, cnli × cnri
    # put the sites together and act with the isometry
    block_1_arr = reshape(composite_tensor_arr, size_cta[1] * size_cta[2], size_cta[3] * size_cta[4])
    block_1_arr = p0 * block_1_arr

    # block 2 is the dag of the above
    # block 2 ordering: cnli × cnri, li × ri
    block_2_arr = adjoint(block_1_arr)

    # putting block 1 and block 2 together!
    # block 1: pull the right neighbors and right index to one side
    block_1_arr = reshape(block_1_arr, size_cta[1], size_cta[2], size_cta[3], size_cta[4])
    block_1_arr = permutedims(block_1_arr, (1, 3, 2, 4))
    block_1_arr = reshape(block_1_arr, size_cta[1] * size_cta[3], size_cta[2] * size_cta[4])

    # block 2: pull the right neighbors and right index to one side
    block_2_arr = reshape(block_2_arr, size_cta[3], size_cta[4], size_cta[1], size_cta[2])
    block_2_arr = permutedims(block_2_arr, (4, 2, 3, 1))
    block_2_arr = reshape(block_2_arr, size_cta[2] * size_cta[4], size_cta[1] * size_cta[3])

    # block_12 ordering: li × cnli, li × cnli
    block_12_arr = block_1_arr * block_2_arr

    # block 3 is compsite_tensor_arr
    # block 3 ordering:  li × ri × cnli, cnri
    block_3_arr = reshape(composite_tensor_arr, size_cta[1] * size_cta[2] * size_cta[3], size_cta[4])

    # block 4 ordering: li × ri, cnli × cnri
    # put the sites together and act with the isometry
    block_4_arr = reshape(composite_tensor_arr, size_cta[1] * size_cta[2], size_cta[3] * size_cta[4])
    block_4_arr = p0 * block_4_arr
    block_4_arr = adjoint(block_4_arr)

    # block 4 ordering: cnli × cnri, li × ri
    # pull the right neighbors together
    block_4_arr = reshape(block_4_arr, size_cta[3], size_cta[4], size_cta[1], size_cta[2])
    block_4_arr = permutedims(block_4_arr, (2, 3, 4, 1))
    # block 4 ordering: ri, li × ri × cnli
    block_4_arr = reshape(block_4_arr, size_cta[4], size_cta[1] * size_cta[2] * size_cta[3])


    # block 34 ordering: li × ri × cnli, li × ri × cnli
    block_34_arr = block_3_arr * block_4_arr

    # put together block_12 and block_34 (block_12 -> li from block 2, block_34 -> li from block 3)
    # block_12 ordering: li × cnli × li, cnli
    block_12_arr = reshape(block_12_arr, size_cta[1] * size_cta[3] * size_cta[1], size_cta[3])

    # block_34 ordering: li × ri × cnli, li × ri × cnli
    block_34_arr = reshape(block_34_arr, size_cta[1], size_cta[2], size_cta[3], size_cta[1] * size_cta[2] * size_cta[3])
    block_34_arr = permutedims(block_34_arr, (3, 1, 2, 4))
    # block_34 ordering: cnli, li × ri × li × ri × cnli
    block_34_arr = reshape(block_34_arr, size_cta[3], size_cta[1] * size_cta[2] * size_cta[1] * size_cta[2] * size_cta[3])

    # put together block_12 and block_34
    # env_arr ordering: li × cnli × li × li × ri, li, ri, cnli
    env_arr = block_12_arr * block_34_arr
    env_arr = reshape(env_arr, size_cta[1] * size_cta[3] * size_cta[1] * size_cta[1] * size_cta[2], size_cta[1], size_cta[2], size_cta[3])
    env_arr = permutedims(env_arr, (1, 3, 2, 4))
    env_arr = reshape(env_arr, size_cta[1] * size_cta[3] * size_cta[1] * size_cta[1] * size_cta[2], size_cta[2], size_cta[1] * size_cta[3])
    # env_arr ordering: li × cnli, li, li × ri, ri, li × cnli
    env_arr = reshape(env_arr, size_cta[1] * size_cta[3], size_cta[1], size_cta[1] * size_cta[2], size_cta[2], size_cta[1] * size_cta[3])
    env_arr = permutedims(env_arr, (1, 5, 2, 3, 4))
    env_arr = reshape(env_arr, size_cta[1] * size_cta[3] * size_cta[1] * size_cta[3], size_cta[1] * size_cta[1] * size_cta[2] * size_cta[2])

    # for perfoming a delta, create a matrix and reshape it as a vector
    id_mat_arr = Matrix((1.0 + 0.0im)I, size_cta[1] * size_cta[3], size_cta[1] * size_cta[3])
    id_mat_arr = reshape(id_mat_arr, 1, size_cta[1] * size_cta[3] * size_cta[1] * size_cta[3])

    # env_arr ordering: 1 × li (b2) li (b3) × ri (b3) × ri (b4)
    env_block = id_mat_arr * env_arr
    env_block = reshape(env_block, size_cta[1], size_cta[1], size_cta[2], size_cta[2])
    env_block = permutedims(env_block, (1, 4, 2, 3))
    env_block = reshape(env_block, size_cta[1] * size_cta[2], size_cta[1] * size_cta[2])

    # check 1: println(env_block)

    F = LinearAlgebra.svd(env_block)

    ## check 2: println(adjoint(F.U))

    ## check 3: println(F.V)

    ## check 4: println(u_kij_arr)

    if ev_original == true
        u_kij_arr = F.V * adjoint(F.U) #  isometry, VU†!
    else
        u_kij_arr = adjoint(F.U) * F.V
    end

    ### all other seven possibilities are also possible!

    elapsed_time = time() - start_time

    combined_tensor = combiner(left_index, right_index; tags = "comb")
    combined_index = combinedind(combined_tensor)

    # here which index to prime is a bit of ambiguity!
    # the following convention reduces the entropy ...
    u_kij = ITensor(u_kij_arr, combined_index', combined_index)

    return false, u_kij, elapsed_time

end


"""
    isometry_ev_mat_env_tensor(composite_object, left_index, right_index, left_neighbors, right_neighbors;
                               manifold = Stiefel, manifold_field = ℂ, initial_p0 = "RS", p0_val = nothing, kwargs...)

Construct and return the raw SVD factorization (`F`) of the environment tensor matrix blocks
for a composite tensor network object using direct array manipulation.

!!! note
    This is a helper function designed specifically to assist in benchmarking, debugging,
    and comparing different decompositions that arise in the environment based isometry update.
    To be used along with [`isometry_ev_mat_compare_combs`](@ref).

## Arguments
- `composite_object`: Composite `ITensor` object on which the isometry acts.
- `left_index`: Left index out of the two indices on which the isometry acts.
- `right_index`: Right index out of the two indices on which the isometry acts.
- `left_neighbors`: Array of indices representing the left environment.
- `right_neighbors`: Array of indices representing the right environment.

## Keywords
- `manifold = Stiefel`: Manifold constructor for optimization (e.g., `Stiefel`, can also be `Unitary`, `Grassmannian`).
- `manifold_field = ℂ`: Number field type over which the manifold is defined (defaults to Complex, ℂ but can also be Real, ℝ field).
- `initial_p0 = "RS"`: Strategy for initial point generation (`"RS"`, `"RE"`, `"I"`, or `"C"`).
- `p0_val = nothing`: Custom initial matrix value if `initial_p0 = "C"`.
- `kwargs...`: Additional keyword arguments (reserved for compatibility).

## Returns
- `F`: The `SVD` factorization object of the computed environment matrix block.

See also: [`isometry_ev`](@ref), [`isometry_ev_it`](@ref), [`isometry_ts`](@ref)
"""
function isometry_ev_mat_env_tensor(composite_object, left_index, right_index, left_neighbors, right_neighbors; manifold = Stiefel, manifold_field = ℂ, initial_p0 = "RS", p0_val = nothing, kwargs...)
    # always check for the orthogonality center before applying the isometry!

    # generate a random isometry that acts on the clubbed space
    if initial_p0 == "RS"
        M1 = manifold(dim(left_index), dim(left_index), manifold_field)
        M2 = manifold(dim(right_index), dim(right_index), manifold_field)
        p0 = kron(rand(M1), rand(M2))
    elseif initial_p0 == "RE"
        M = manifold(dim(left_index) * dim(right_index), dim(left_index) * dim(right_index), manifold_field)
        p0 = rand(M)
    elseif initial_p0 == "I"
        # id_m = Matrix((1.0+0.0im)I, dim(left_index), dim(right_index))
        # p0 = kron(id_m, id_m)
        p0 = Matrix((1.0 + 0.0im)I, dim(left_index) * dim(right_index), dim(left_index) * dim(right_index))
    elseif initial_p0 == "C" && p0_val != nothing
        p0 = p0_val
    end

    combined_ln_tensor = nothing
    combined_rn_tensor = nothing

    if length(left_neighbors) != 0
        combined_ln_tensor = combiner(left_neighbors; tags = "ln_comb")
        combined_ln_index = combinedind(combined_ln_tensor)
    end

    # combine all the right neighbors into a single index
    if length(right_neighbors) != 0
        combined_rn_tensor = combiner(right_neighbors; tags = "rn_comb")
        combined_rn_index = combinedind(combined_rn_tensor)
    end

    composite_tensor = deepcopy(composite_object)

    if combined_ln_tensor != nothing
        composite_tensor = composite_tensor * combined_ln_tensor
    end
    if combined_rn_tensor != nothing
        composite_tensor = composite_tensor * combined_rn_tensor
    end


    dim_left_index = dim(left_index)
    dim_right_index = dim(right_index)

    if combined_ln_tensor != nothing && combined_rn_tensor != nothing
        composite_tensor = permute(composite_tensor, left_index, right_index, combined_ln_index, combined_rn_index)
        composite_tensor_arr = Array(composite_tensor.tensor)

    elseif combined_ln_tensor != nothing
        composite_tensor = permute(composite_tensor, left_index, right_index, combined_ln_index)
        composite_tensor_arr = Array(composite_tensor.tensor)
        composite_tensor_arr = reshape(composite_tensor_p_arr, dim(left_index), dim(right_index), dim(combined_ln_index), 1)

    elseif combined_rn_tensor != nothing
        composite_tensor = permute(composite_tensor, left_index, right_index, combined_rn_index)
        composite_tensor_arr = Array(composite_tensor.tensor)
        composite_tensor_arr = reshape(composite_tensor_arr, dim(left_index), dim(right_index), 1, dim(combined_rn_index))
    end

    #####################

    # composite_tensor_arr has ordering: left_index, right_index, prod(left_neighbor_arr), prod(right_neighbor_arr)
    size_cta = size(composite_tensor_arr)

    # we can think of the algorithm as four different blocks
    # block 1 ordering: li × ri, cnli × cnri
    # put the sites together and act with the isometry
    block_1_arr = reshape(composite_tensor_arr, size_cta[1] * size_cta[2], size_cta[3] * size_cta[4])
    block_1_arr = p0 * block_1_arr

    # block 2 is the dag of the above
    # block 2 ordering: cnli × cnri, li × ri
    block_2_arr = adjoint(block_1_arr)

    # putting block 1 and block 2 together!
    # block 1: pull the right neighbors and right index to one side
    block_1_arr = reshape(block_1_arr, size_cta[1], size_cta[2], size_cta[3], size_cta[4])
    block_1_arr = permutedims(block_1_arr, (1, 3, 2, 4))
    block_1_arr = reshape(block_1_arr, size_cta[1] * size_cta[3], size_cta[2] * size_cta[4])

    # block 2: pull the right neighbors and right index to one side
    block_2_arr = reshape(block_2_arr, size_cta[3], size_cta[4], size_cta[1], size_cta[2])
    block_2_arr = permutedims(block_2_arr, (4, 2, 3, 1))
    block_2_arr = reshape(block_2_arr, size_cta[2] * size_cta[4], size_cta[1] * size_cta[3])

    # block_12 ordering: li × cnli, li × cnli
    block_12_arr = block_1_arr * block_2_arr

    # block 3 is compsite_tensor_arr
    # block 3 ordering:  li × ri × cnli, cnri
    block_3_arr = reshape(composite_tensor_arr, size_cta[1] * size_cta[2] * size_cta[3], size_cta[4])

    # block 4 ordering: li × ri, cnli × cnri
    # put the sites together and act with the isometry
    block_4_arr = reshape(composite_tensor_arr, size_cta[1] * size_cta[2], size_cta[3] * size_cta[4])
    block_4_arr = p0 * block_4_arr
    block_4_arr = adjoint(block_4_arr)

    # block 4 ordering: cnli × cnri, li × ri
    # pull the right neighbors together
    block_4_arr = reshape(block_4_arr, size_cta[3], size_cta[4], size_cta[1], size_cta[2])
    block_4_arr = permutedims(block_4_arr, (2, 3, 4, 1))
    # block 4 ordering: ri, li × ri × cnli
    block_4_arr = reshape(block_4_arr, size_cta[4], size_cta[1] * size_cta[2] * size_cta[3])


    # block 34 ordering: li × ri × cnli, li × ri × cnli
    block_34_arr = block_3_arr * block_4_arr

    # put together block_12 and block_34 (block_12 -> li from block 2, block_34 -> li from block 3)
    # block_12 ordering: li × cnli × li, cnli
    block_12_arr = reshape(block_12_arr, size_cta[1] * size_cta[3] * size_cta[1], size_cta[3])

    # block_34 ordering: li × ri × cnli, li × ri × cnli
    block_34_arr = reshape(block_34_arr, size_cta[1], size_cta[2], size_cta[3], size_cta[1] * size_cta[2] * size_cta[3])
    block_34_arr = permutedims(block_34_arr, (3, 1, 2, 4))
    # block_34 ordering: cnli, li × ri × li × ri × cnli
    block_34_arr = reshape(block_34_arr, size_cta[3], size_cta[1] * size_cta[2] * size_cta[1] * size_cta[2] * size_cta[3])

    # put together block_12 and block_34
    # env_arr ordering: li × cnli × li × li × ri, li, ri, cnli
    env_arr = block_12_arr * block_34_arr
    env_arr = reshape(env_arr, size_cta[1] * size_cta[3] * size_cta[1] * size_cta[1] * size_cta[2], size_cta[1], size_cta[2], size_cta[3])
    env_arr = permutedims(env_arr, (1, 3, 2, 4))
    env_arr = reshape(env_arr, size_cta[1] * size_cta[3] * size_cta[1] * size_cta[1] * size_cta[2], size_cta[2], size_cta[1] * size_cta[3])
    # env_arr ordering: li × cnli, li, li × ri, ri, li × cnli
    env_arr = reshape(env_arr, size_cta[1] * size_cta[3], size_cta[1], size_cta[1] * size_cta[2], size_cta[2], size_cta[1] * size_cta[3])
    env_arr = permutedims(env_arr, (1, 5, 2, 3, 4))
    env_arr = reshape(env_arr, size_cta[1] * size_cta[3] * size_cta[1] * size_cta[3], size_cta[1] * size_cta[1] * size_cta[2] * size_cta[2])

    # for perfoming a delta, create a matrix and reshape it as a vector
    id_mat_arr = Matrix((1.0 + 0.0im)I, size_cta[1] * size_cta[3], size_cta[1] * size_cta[3])
    id_mat_arr = reshape(id_mat_arr, 1, size_cta[1] * size_cta[3] * size_cta[1] * size_cta[3])

    # env_arr ordering: 1 × li (b2) li (b3) × ri (b3) × ri (b4)
    env_block = id_mat_arr * env_arr
    env_block = reshape(env_block, size_cta[1], size_cta[1], size_cta[2], size_cta[2])
    env_block = permutedims(env_block, (1, 4, 2, 3))
    env_block = reshape(env_block, size_cta[1] * size_cta[2], size_cta[1] * size_cta[2])

    F = LinearAlgebra.svd(env_block)

    return F

end

"""
    isometry_ev_mat_compare_combs(svd_object, left_index, right_index, left_neighbors,
                                  right_neighbors, leading_u, leading_adjoint,
                                  trailing_adjoint; kwargs...)

Combine and permute the U and V factors from an SVD factorization object using specified
flag configurations to test and compare alternative isometric constructions that disentangle
a specific network.

!!! note
    This is a helper utility designed to systematically test different algebraic choices
    for the construction of the isometry from the SVD of the environment tensor. To be used
    along with [`isometry_ev_mat_env_tensor`](@ref).

## Arguments
- `svd_object`: The `SVD` factorization of the environment (containing `.U` and `.V` fields).
- `left_index`: Left index out of the two indices on which the isometry acts.
- `right_index`: Right index out of the two indices on which the isometry acts.
- `left_neighbors`: Array of indices representing the left environment.
- `right_neighbors`: Array of indices representing the right environment.
- `leading_u`: Boolean flag; if `true`, uses `svd_object.U` as the leading factor (`false` uses `svd_object.V`).
- `leading_adjoint`: Boolean flag; if `true`, applies the adjoint to the leading factor matrix.
- `trailing_adjoint`: Boolean flag; if `true`, applies the adjoint to the trailing factor matrix.

## Keywords
- `kwargs...`: Additional keyword arguments (reserved for compatibility).

## Returns
- `u_kij`: Computed isometry as an ITensor with primed and unprimed combined indices.

See also: [`isometry_ev_mat_env_tensor`](@ref), [`isometry_ev`](@ref)
"""
function isometry_ev_mat_compare_combs(svd_object, left_index, right_index, left_neighbors, right_neighbors, leading_u, leading_adjoint, trailing_adjoint; kwargs...)

    if leading_u == true
        leading_iso = svd_object.U
        trailing_iso = svd_object.V
    else
        leading_iso = svd_object.V
        trailing_iso = svd_object.U
    end

    if leading_adjoint == true
        leading_iso = adjoint(leading_iso)
    end
    if trailing_adjoint == true
        trailing_iso = adjoint(trailing_iso)
    end

    u_kij_arr = leading_iso * trailing_iso

    # @assert u_kij_arr*adjoint(u_kij_arr) == Matrix((1.0+0.0im)*I, size(leading_iso)[1], size(leading_iso)[2])

    ### all other seven possibilities are also possible!

    combined_tensor = combiner(left_index, right_index; tags = "comb")
    combined_index = combinedind(combined_tensor)

    # here which index to prime is a bit of ambiguity!
    # the following convention reduces the entropy ...
    u_kij = ITensor(u_kij_arr, combined_index', combined_index)

    return u_kij

end


export isometry_ev_it, isometry_ev, isometry_ev_mat_env_tensor, isometry_ev_mat_compare_combs
