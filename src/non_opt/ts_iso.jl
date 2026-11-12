using .TensorSketch

"""
    isometry_ts(composite_object, left_index, right_index, left_neighbors, right_neighbors; kwargs...)

Compute an optimal isometry using a fast disentangling algorithm (`fastDisentangle` implemented as in [1])
on a composite tensor network object across specified cuts.

!!! note
    If the fast disentangling routine throws an exception during execution,
    this function falls back gracefully by returning an identity matrix for the isometry
    along with an `error_loop` flag set to `true`.

!!! note
    This function is designed to be used in conjunction with [`apply_optimized_isometry`](@ref) in order
    to return to the original form. See examples (add reference to the example).

## Arguments
- `composite_object`: Composite ITensor object on which the isometry acts.
- `left_index`: Left index out of the two indices on which the isometry acts.
- `right_index`: Right index out of the two indices on which the isometry acts.
- `left_neighbors`: Array of indices representing the left environment.
- `right_neighbors`: Array of indices representing the right environment.

## Keywords
- `kwargs...`: Additional keyword arguments passed to internal tensor operations or solvers.

## Returns
- `error_loop`: A boolean flag indicating whether a fallback error recovery occurred (`true` if an error was caught, `false` otherwise).
- `u_kij`: Computed isometry as an `ITensor` with primed and unprimed combined indices.
- `elapsed_time`: Total execution time in seconds, or `nothing` if an error occurred.

## References
This implementation is integrated into the package (as a `TensorSketch` module) from the following code repo as cited in the original paper:
Kevin Slagle (Year). *Fast Tensor Disentangling Algorithm, SciPost Phys. 11, 056 (2021).*
[1] [fastDisentangle](https://github.com/kjslag/fastDisentangle).

See also: [`isometry_ro_are`](@ref), [`isometry_ro_sre`](@ref)
"""
function isometry_ts(composite_object, left_index, right_index, left_neighbors, right_neighbors; kwargs...)

    # combine the indices on which the isometry acts
    combined_iso_tensor = combiner(left_index, right_index; tags = "iso_comb")
    combined_iso_index = combinedind(combined_iso_tensor)

    combined_ln_tensor = nothing
    combined_rn_tensor = nothing

    # combine all the left neighbors into a single index
    if length(left_neighbors) != 0
        combined_ln_tensor = combiner(left_neighbors; tags = "ln_comb")
        combined_ln_index = combinedind(combined_ln_tensor)
    end

    # combine all the right neighbors into a single index
    if length(right_neighbors) != 0
        combined_rn_tensor = combiner(right_neighbors; tags = "rn_comb")
        combined_rn_index = combinedind(combined_rn_tensor)
    end

    # construct the composite tensor with the combined indices
    composite_tensor = composite_object * combined_iso_tensor

    if combined_ln_tensor != nothing
        composite_tensor = composite_tensor * combined_ln_tensor
    end
    if combined_rn_tensor != nothing
        composite_tensor = composite_tensor * combined_rn_tensor
    end

    dim_left_index = dim(left_index)
    dim_right_index = dim(right_index)

    if combined_ln_tensor != nothing && combined_rn_tensor != nothing
        composite_tensor_p = permute(composite_tensor, combined_iso_index, combined_ln_index, combined_rn_index)
        composite_tensor_p_arr = Array(composite_tensor_p.tensor)
    elseif combined_ln_tensor != nothing
        composite_tensor_p = permute(composite_tensor, combined_iso_index, combined_ln_index)
        composite_tensor_p_arr = Array(composite_tensor_p.tensor)
        composite_tensor_p_arr = reshape(composite_tensor_p_arr, dim(combined_iso_index), dim(combined_ln_index), 1)
    elseif combined_rn_tensor != nothing
        composite_tensor_p = permute(composite_tensor, combined_iso_index, combined_rn_index)
        composite_tensor_p_arr = Array(composite_tensor_p.tensor)
        composite_tensor_p_arr = reshape(composite_tensor_p_arr, dim(combined_iso_index), 1, dim(combined_rn_index))
    end

    error_loop = false

    try

        start_time = time()

        U = fastDisentangle(dim_left_index, dim_right_index, composite_tensor_p_arr)

        elapsed_time = time() - start_time

        U_re = reshape(U, dim_left_index * dim_right_index, dim_left_index * dim_right_index)

        u_kij = ITensor(U_re, combined_iso_index', combined_iso_index)

        return error_loop, u_kij, elapsed_time

        # we could have returned none! but we return identity to maintain the workflow (error_loop will give us the necessary info!)
    catch err

        error_loop = true

        elapsed_time = nothing

        U_re = Matrix(1.0I, dim_left_index * dim_right_index, dim_left_index * dim_right_index)

        u_kij = ITensor(U_re, combined_iso_index', combined_iso_index)

        return error_loop, u_kij, elapsed_time

    end


end

export isometry_ts
