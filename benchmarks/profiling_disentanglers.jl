using ITensors
using ITensorMPS
using Disentanglers
using Manifolds
using Manopt
using LinearAlgebra

# using Random -> to generate the seed

## Data generation functions!
"""
    generate_random_isometries(single_index_dim, iso_id)

Generate a random complex isometry on the Stiefel manifold and save it to an HDF5 file.

The function creates a random point on the complex Stiefel manifold ``\text{St}(d^2, d^2, ℂ)``
where ``d`` is dimension of the single index (`single_index_dim`), the isometry acts on ``d^2`` and writes the
resulting matrix to an HDF5 file.

# Arguments
- `single_index_dim::Integer`: The single index dimension (``d``). The manifold dimension will be ``(d^2 × d^2)``.
- `iso_id`: An identifier or index used to uniquely name the output HDF5 file.

# Returns
- HDF5 file that holds the random isometry.

# Examples
```julia
# Generate an isometry with single index dimension 2 and identifier 1
Disentanglers.generate_random_isometries(2, 1)
# This creates a file named "l1_2_r1_2_id_1.h5" containing the isometry "rand_iso".
```
"""
function generate_random_isometries(single_index_dim, iso_id)

    M = Stiefel(single_index_dim * single_index_dim, single_index_dim * single_index_dim, ℂ)

    fn = "l1_$(single_index_dim)_r1_$(single_index_dim)_id_$(iso_id).h5"
    rand_iso = rand(M)

    ## write the random isometry, a matrix to .h5 file
    return h5open(fn, "w") do file
        write(file, "rand_iso", rand_iso)
    end

end

"""
    random_normalized_four_index_itensors(ind_l1, ind_l2, ind_r1, ind_r2, len_sv)

Generate a random, normalized four-index `ITensor` with specified length of the singular values.

This function constructs a random complex matrix by generating semi-orthogonal matrices via
QR decomposition and scaling them with a normalized random singular value spectrum of length `len_sv`.
The resulting matrix is then reshaped and wrapped into an `ITensor` with the provided indices.

# Arguments
- `ind_l1`: The first left `Index` of the ITensor.
- `ind_l2`: The second left `Index` of the ITensor.
- `ind_r1`: The first right `Index` of the ITensor.
- `ind_r2`: The second right `Index` of the ITensor.
- `len_sv::Integer`: The number of singular values (and rank) to use, which must be less than or equal to the minimum of the combined left and right dimensions.

# Returns
- `ITensor`: A four-index `ITensor` spanning `ind_l1`, `ind_l2`, `ind_r1`, and `ind_r2`.

# Throws
- Throws an error if `len_sv` exceeds the minimum dimension of the combined left or right spaces (min(m, n)), where m=dim(`ind_l1`) × dim(`ind_l2`), n=dim(`ind_r1`) × dim(`ind_r2`).

# Examples
```julia
using ITensors

l1 = Index(2, "l1")
l2 = Index(2, "l2")
r1 = Index(2, "r1")
r2 = Index(2, "r2")

# Create a random normalized four-index ITensor with a singular value spectrum of length 2
T = Disentanglers.random_normalized_four_index_itensors(l1, l2, r1, r2, 2)
```
"""
function random_normalized_four_index_itensors(ind_l1, ind_l2, ind_r1, ind_r2, len_sv)

    # put left indices together: dim(ind_l1)*dim(ind_l2)
    m = dim(ind_l1) * dim(ind_l2)
    # put right indices together: dim(ind_r1)*dim(ind_r2)
    n = dim(ind_r1) * dim(ind_r2)

    if len_sv > minimum([m, n])

        return error()

    else

        k = len_sv

        U = Matrix(qr(randn(ComplexF64, m, k)).Q)
        V = Matrix(qr(randn(ComplexF64, n, k)).Q)

        rmat = Matrix(U) * diagm(normalize!(rand(len_sv))) * Matrix(V)'

        rmat = reshape(rmat, dim(ind_l1), dim(ind_l2), dim(ind_r1), dim(ind_r2))
        rmat_it = ITensor(rmat, ind_l1, ind_l2, ind_r1, ind_r2)

        return rmat_it

    end


end

"""
    check_and_save_rnit(dim_left_index, dim_left_neighbor, dim_right_index, dim_right_neighbor, sv_len, fn)

Create, validate, and save a random normalized four-index ITensor to an HDF5 file.

This function constructs four ITensor `Index` objects with specific dimensions and tags,
generates a random normalized four-index ITensor via [`random_normalized_four_index_itensors`](@ref),
performs a singular value decomposition (SVD) check, and exports the resulting tensor to an HDF5 file.

# Arguments
- `dim_left_index::Integer`: Dimension for the left index on which the isometry acts.
- `dim_left_neighbor::Integer`: Dimension for the left neighbor.
- `dim_right_index::Integer`: Dimension for the right index on which the isometry acts.
- `dim_right_neighbor::Integer`: Dimension for the right neighbor.
- `sv_len::Integer`: The number of singular values / rank for the random ITensor. Must not exceed the minimum of the combined left and right dimensions.
- `fn::AbstractString`: The filename or file path for the output HDF5 file.

# Returns
- HDF5 file with random normalized four-index ITensor, or an error message string `String` if an exception occurs (e.g., if `sv_len` exceeds the valid dimension bounds).

# Examples
```julia
# Create indices with dimensions 2, 2, 2, 2, a singular value length of 2,
# and save the resulting ITensor to "random_tensor.h5"
Disentanglers.check_and_save_rnit(2, 2, 2, 2, 2, "random_tensor.h5")
```
"""
function check_and_save_rnit(dim_left_index, dim_left_neighbor, dim_right_index, dim_right_neighbor, sv_len, fn)

    il1 = Index(dim_left_index, "i")
    il2 = Index(dim_left_neighbor, "k1")

    ir1 = Index(dim_right_index, "j")
    ir2 = Index(dim_right_neighbor, "k2")

    try

        rmat = random_normalized_four_index_itensors(il1, il2, ir1, ir2, sv_len)

        u, s, v = svd(rmat, (il1, il2))

        # check norm as a test
        # norm(diag(Array(s.tensor)))

        # if normalized, save the random normalized ITensor to `fn` file
        h5open(fn, "w") do file
            write(file, "rand_flt", rmat)
        end

    catch e

        # we might not get here, but just in case!
        return "Error: Choose singular value length to be no greater [combined left, combined right]"

    end

end

"""
    generate_many_rn_itensors(ntraj)

Generate and save a batch of random normalized four-index ITensors across a grid of dimensions and bond dimensions.

This function iterates through a predefined grid of left index dimensions (`[2, 4, 8, 16]`) and maximum bond dimensions (``χ_{max}``, `[32, 64, 128, 256, 512]`).
For each parameter combination, it generates `ntraj` random normalized ITensors using [`check_and_save_rnit`](@ref) and saves each to a uniquely named HDF5 file.

# Arguments
- `ntraj::Integer`: The number of trajectories (or individual random tensors) to generate for each parameter configuration in the grid.

# Returns
- `nothing`: The function executes for its side effects (saving HDF5 files).

# Examples
```julia
# Generate 5 trajectories for each dimension and bond dimension combination
Disentanglers.generate_many_rn_itensors(5)
```
"""
function generate_many_rn_itensors(ntraj)

    # left index dim arr
    l_dim_arr = [2, 4, 8, 16]

    # χ_max dim arr
    chi_max_arr = [32, 64, 128, 256, 512]

    l_chi_arr = [[x, y] for x in l_dim_arr for y in chi_max_arr]

    for a_ind in 1:length(l_chi_arr)

        l = l_chi_arr[a_ind][1]
        k = Int(l_chi_arr[a_ind][2] / l)

        for rnit_id in 1:ntraj

            l1 = l
            l2 = k
            r1 = l
            r2 = k
            sv = l_chi_arr[a_ind][2]

            fn = "l1_$(l1)_l2_$(l2)_r1_$(r1)_r2_$(r2)_chi_max_$(sv)_id_$(rnit_id).h5"
            check_and_save_rnit(l1, l2, r1, r2, sv, fn)

        end

    end

    return
end

"""
    benchmark_ro_opt(opt_function, left_index_dim, left_neighbor_dim, right_index_dim, right_neighbor_dim, sv, rnit_id, iso_id, solver; kwargs...)

Benchmark an Riemannian manifold based optimization routine for disentangling a four-legged ITensor using pre-saved random tensors and initializations.

This function loads a target four-legged tensor (structured as `[left_index, left_neighbor, right_index, right_neighbor]`)
and an initial guess isometry from HDF5 files based on the specified dimensions and identifiers. It then executes
the provided optimization function (`opt_function`) using the chosen solver and records performance metrics.

# Arguments
- `opt_function`: The optimization function to execute (e.g., `isometry_ro_are`, `isometry_ro_sre`).
- `left_index_dim::Integer`: Dimension of the left physical index (on which the isometry acts).
- `left_neighbor_dim::Integer`: Dimension of the left neighbor index.
- `right_index_dim::Integer`: Dimension of the right physical index (on which the isometry acts).
- `right_neighbor_dim::Integer`: Dimension of the right neighbor index.
- `sv::Integer`: Maximum bond dimension (``χ_{\text{max}}``) or singular value threshold used in the filename lookup.
- `rnit_id::Integer`: Trajectory identifier for the random normalized `ITensor` file.
- `iso_id::Integer`: Identifier for the initial isometry file.
- `solver`: Optimization solver method. Options typically include `NelderMead`, `particle_swarm`, `gradient_descent`, `conjugate_gradient_descent`, `quasi_Newton`, `subgradient_method`, `adaptive_regularization_with_cubics`, `trust_regions`, or `stochastic_gradient_descent`.
- `kwargs...`: Additional keyword arguments passed directly to `opt_function` (such as `gradient_free_solver=true`).

# Returns
- A tuple containing:
  - `rand_it`: The loaded initial `ITensor` to be disentangled, in place disentangled
  - `opt_iso`: The optimized isometry result.
  - `icf`: Initial cost function value.
  - `fcf`: Final cost function value.
  - `elapsed_time`: Total execution time for the optimization.

# Examples
```julia
# Run a benchmark optimization using isometry_ro_are and NelderMead
r_it, iso_opt, cfi, cff, time_elapsed = Disentanglers.benchmark_ro_opt(
    isometry_ro_are, 2, 16, 2, 16, 32, 1, 1, NelderMead;
    gradient_free_solver = true
)
```
"""
function benchmark_ro_opt(opt_function, left_index_dim, left_neighbor_dim, right_index_dim, right_neighbor_dim, sv, rnit_id, iso_id, solver; kwargs...)

    fn = "l1_$(left_index_dim)_l2_$(left_neighbor_dim)_r1_$(right_index_dim)_r2_$(right_neighbor_dim)_chi_max_$(sv)_id_$(rnit_id).h5"

    rand_it = nothing

    # load the initial tensor to disentangle!
    h5open(fn, "r") do file0
        rand_it = read(file0, "rand_flt", ITensor)
    end

    inds_rand_it = inds(rand_it)

    record_items = [:Iteration, :Cost]

    # load the initialization!
    fn_p0 = "l1_$(left_index_dim)_r1_$(right_index_dim)_id_$(iso_id).h5"
    init_p0 = nothing

    h5open(fn_p0, "r") do file1
        init_p0 = read(file1, "rand_iso")
    end

    opt_iso, res_obj, icf, fcf, elapsed_time = opt_function(rand_it, inds_rand_it[1], inds_rand_it[3], [inds_rand_it[2]], [inds_rand_it[4]]; initial_p0 = "C", p0_val = init_p0, record = record_items, opt = solver, kwargs...)

    n_iters = get_record(res_obj, :Iteration, :Iteration)
    cf_iters = get_record(res_obj, :Iteration, :Cost)

    return rand_it, opt_iso, icf, fcf, elapsed_time

end

### ----- ###

"""
    benchmark_non_ro_opt(opt_function, left_index_dim, left_neighbor_dim, right_index_dim, right_neighbor_dim, sv, rnit_id, iso_id; kwargs...)

Benchmark a non-Riemannian-optimized routine for a four-legged ITensor using pre-saved random tensors and initializations.

This function loads a target four-legged ITensor (structured as `[left_index, left_neighbor, right_index, right_neighbor]`) and an initial guess isometry from HDF5 files.
It then executes the specified non-Riemannian optimization routine (`opt_function`) and tracks the error output, optimized isometry, and elapsed execution time.

# Arguments
- `opt_function`: The non-Riemannian optimization function to execute (e.g., `isometry_ts`, `isometry_ev`)(either tensor sketching or Evenbly-Vidal)
- `left_index_dim::Integer`: Dimension of the left physical index (on which the isometry acts).
- `left_neighbor_dim::Integer`: Dimension of the left neighbor index.
- `right_index_dim::Integer`: Dimension of the right physical index (on which the isometry acts).
- `right_neighbor_dim::Integer`: Dimension of the right neighbor index.
- `sv::Integer`: Maximum bond dimension (``χ_{\text{max}}``) or singular value threshold used in the filename lookup.
- `rnit_id::Integer`: Trajectory identifier for the random normalized `ITensor` file.
- `iso_id::Integer`: Identifier for the initial isometry file.
- `kwargs...`: Additional keyword arguments passed directly to `opt_function` (such as `ev_original = false`).

# Returns
- A tuple containing:
  - `rand_it`: The loaded initial `ITensor` to be processed.
  - `error_output`: The error ouput in case the method fails to be find an isometry.
  - `opt_iso`: The resulting optimized isometry.
  - `elapsed_time`: Total execution time for the routine.

# Examples
```julia
# Run a benchmark using a non-ro optimization function like isometry_ts
r_it, err_out, iso_opt, time_elapsed = Disentanglers.benchmark_non_ro_opt(
    isometry_ts, 2, 16, 2, 16, 32, 1, 1;
    ev_original = false
)
```
"""
function benchmark_non_ro_opt(opt_function, left_index_dim, left_neighbor_dim, right_index_dim, right_neighbor_dim, sv, rnit_id, iso_id; kwargs...)

    fn = "l1_$(left_index_dim)_l2_$(left_neighbor_dim)_r1_$(right_index_dim)_r2_$(right_neighbor_dim)_chim_$(sv)_id_$(rnit_id).h5"

    rand_it = nothing

    # load the initial tensor to disentangle!
    h5open(fn, "r") do file0
        rand_it = read(file0, "rand_flt", ITensor)
    end

    inds_rand_it = inds(rand_it)

    # load the initialization!
    fn_p0 = "l1_$(left_index_dim)_r1_$(right_index_dim)_id_$(iso_id).h5"
    init_p0 = nothing

    h5open(fn_p0, "r") do file1
        init_p0 = read(file1, "rand_iso")
    end

    error_output, opt_iso, elapsed_time = opt_function(rand_it, inds_rand_it[1], inds_rand_it[3], [inds_rand_it[2]], [inds_rand_it[4]]; initial_p0 = "C", p0_val = init_p0, kwargs...)

    return rand_it, error_output, opt_iso, elapsed_time

end
