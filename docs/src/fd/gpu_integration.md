# [GPU integration](@id gpu_integration)

Here, we outline different strategies that enable GPU compute. One way to enable heterogeneous compute is to generalize the current functionality to use [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). For now, we integrate with CUDA to run on Nvidia GPUs. The branch `gpu` has been adapted to run using CUDA on Nvidia GPUs. We list the changes that allows for the integration of the solvers. We focus on integrating solvers from `Manopt.jl` while other solvers can be similarly incorporated.

### Adaptions

Considering the entropy based objective functions as an example, we present the adaptions that need to be made for successful runs on GPU. We note that only few solvers (and specific options thereof) are currently supported by `Manopt.jl` to function on GPUs.

#### Objective function modifications

In `second_renyi_entropy` modify the return to avoid `Scalar indexing is disallowed` error:

```julia
# Modify the following
return -log(real(tr(rho_sq_contract)))

# to the following equivalent form
return -log(real(sum(rho_contract .* rho_contract')))
```


Further, modify the function `isometry_ro_sre` signature to include the `cuda_flag` option

```julia
# Modify the function signature
function isometry_ro_sre(composite_object, left_index, right_index, left_neighbors, right_neighbors; gradient_free_solver = false, manifold = Stiefel, manifold_field = ℂ, opt = quasi_Newton, initial_p0 = "RS", p0_val = nothing, kwargs...)

# to the following equivalent signature
function isometry_ro_sre(composite_object, left_index, right_index, left_neighbors, right_neighbors; gradient_free_solver = false, manifold = Stiefel, manifold_field = ℂ, opt = quasi_Newton, initial_p0 = "RS", cuda_flag = false, p0_val = nothing, kwargs...)
```

Turn the composite tensor type on which the isometry acts into a `CUDA` type using the `cuda_flag`, adapt fixes the precision else this defaults to single float

```julia
if cuda_flag == true
    composite_tensor_arr = adapt(CuArray, composite_tensor_arr)
end
```

Turn the initialization to the optimizer into a CUDA
```julia
if cuda_flag == true
    p0 = adapt(CuArray, p0)
    p_id = adapt(CuArray, p_id)
end
```

This sets up for running the optimization on GPUs, the following is an example for disentangling a random normalized four leg tensor using GPUs

```julia
using Disentanglers
using ITensors
using ITensorMPS
using Manifolds
using Manopt

# generate random tensor
l1 = Index(8, "l1")
l2 = Index(16, "l2")
r1 = Index(8, "r1")
r2 = Index(16, "r2")

# Create a random normalized four-index ITensor with a singular value spectrum of length 2
T = Disentanglers.random_normalized_four_index_itensors(l1, l2, r1, r2, 32)

println(inds(T))

# using the second-Renyi entropy as the objective function
u_kij, res, initial_objective_val, final_objective_val, elapsed_time = isometry_ro_sre(T, l1, r1, [l2], [r2]; opt = gradient_descent, cuda_flag = true, stepsize = ConstantLength(0.1))

# using the α-Renyi entropy as the objective function -> the code on the branch `gpu` has been updated accordingly
u_kij, res, initial_objective_val, final_objective_val, elapsed_time = isometry_ro_are(T, l1, r1, [l2], [r2]; opt = gradient_descent, cuda_flag = true, stepsize = ConstantLength(0.1))

println(u_kij)
println(res)
println(initial_objective_val)
println(final_objective_val)
println(elapsed_time)

# apply_optimized_isometry(opt_isometry, composite_object, left_index, right_index, left_neighbors, right_neighbors)
```
