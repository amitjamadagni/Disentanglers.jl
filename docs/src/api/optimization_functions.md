# [Optimization Functions](@id OptimizingFuncs)
We list the different optimization routines that appear in the package.

---

###### [Riemannian based disentanglers](@id RbD)
* [`Disentanglers.isometry_ro_are`](@ref isometry_ro_are)
* [`Disentanglers.isometry_ro_sre`](@ref isometry_ro_sre)
* [`Disentanglers.isometry_unitary_overlap`](@ref isometry_unitary_overlap)


###### [Tensor Sketching based disentangler](@id TsbD)
* [`Disentanglers.isometry_ts`](@ref isometry_ts)

###### [Environment based disentangler](@id EbD)
* [`Disentanglers.isometry_ev`](@ref isometry_ev)
* [`Disentanglers.isometry_ev_it`](@ref isometry_ev_it)
* [`Disentanglers.isometry_ev_mat_env_tensor`](@ref isometry_ev_mat_env_tensor)
* [`Disentanglers.isometry_ev_mat_compare_combs`](@ref isometry_ev_mat_compare_combs)

## Riemannian based disentanglers
```@docs
Disentanglers.isometry_ro_are
Disentanglers.isometry_ro_sre
Disentanglers.isometry_unitary_overlap
```

## Tensor Sketching based disentangler
```@docs
Disentanglers.isometry_ts
```

## Environment based disentangler
```@docs
Disentanglers.isometry_ev
Disentanglers.isometry_ev_it
Disentanglers.isometry_ev_mat_env_tensor
Disentanglers.isometry_ev_mat_compare_combs
```
