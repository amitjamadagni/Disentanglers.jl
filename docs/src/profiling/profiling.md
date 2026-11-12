# [Profiling the different disentanglers](@id DisentanglingPerf)

We present functions that profile the performance of different disentangling algorithms. We also document the different API calls used to generate the random tensors and the associated isometric initializations, see [generating data](@ref gen_data) below.

### [Benchmark functions](@id bench_funcs)

#### Wrapper to benchmark the Riemannian disentanglers

The function [`Disentanglers.benchmark_ro_opt`](@ref Disentanglers.benchmark_ro_opt) can be used to profile the performance of different Riemannian disentanglers. It allows for the setting of the objective function to be either the α-Rényi entropy, see [`Disentanglers.isometry_ro_are`](@ref isometry_ro_are) or to be second-Rényi entropy, see [`Disentanglers.isometry_ro_sre`](@ref isometry_ro_sre). For each of the objective function, we can further choose different solvers from `Manopt.jl`.

```@docs
Disentanglers.benchmark_ro_opt
```

#### Wrapper to benchmark the non-Riemannian disentanglers

The function [`Disentanglers.benchmark_non_ro_opt`](@ref Disentanglers.benchmark_non_ro_opt) can be used to profile the performance of tensor sketch, see [`Disentanglers.isometry_ts`](@ref isometry_ts) and Evenbly-Vidal environment based, see [`Disentanglers.isometry_ev`](@ref isometry_ev) isometric updates that minimize the entropy associated with the virtual index.

```@docs
Disentanglers.benchmark_non_ro_opt
```

### [API calls](@id apicalls)

#### [Generating data: Random isometries to random tensors](@id gen_data)
- Generating random isometries of fixed size
```@docs
Disentanglers.generate_random_isometries
```

- Generating random normalized four index `ITensor`
```@docs
Disentanglers.random_normalized_four_index_itensors
```

- Validate the normalization of singular values and save the random normalized four index `ITensor`
```@docs
Disentanglers.check_and_save_rnit
```

- Generate many instances of the random normalized tensors
```@docs
Disentanglers.generate_many_rn_itensors
```
