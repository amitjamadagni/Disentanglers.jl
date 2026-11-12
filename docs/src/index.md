# Disentanglers

```@raw html
<div align="center">
  <br>
  <img src="./assets/logo.gif" alt="Disentanglers.jl" width="200">
  <br>
</div>
&nbsp;
```

Julia package with a collection of tools to optimize tensor network representations. The central theme is to optimize representational entropy which might in some cases lead to compressed representations but compression is not always guaranteed. We build on ITensors.jl and integrate various solvers from Manopt.jl (manifold optimization) leading to Riemannian manifold based disentanglers, tensor sketching based disentanglers and environment based disentanglers. The package provides routines that profile the performance of these methods. Furthermore, the applications include circuit compilation for Matrix Product State (MPS), Matrix Product Operator (MPO) and optimizing the representation of Locally Purified Density Operators (LPDOs).


## Installation

- Clone the package

```
git clone https://code.ornl.gov/cgg/Disentanglers.jl.git
```

- Install Disentanglers.jl via the Julia REPL:
```julia
julia> using Pkg
julia> Pkg.add("path/to/Disentanglers.jl");
```
or
```julia
julia> ] add path/to/Disentanglers.jl
```

## Dependencies

The package is built on the following packages
- [ITensors.jl](https://github.com/ITensor/ITensors.jl),
- [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl),
- [Manifolds.jl](https://github.com/JuliaManifolds/Manifolds.jl),
- [Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl),
- [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)(mainly [Zygote.jl](https://github.com/FluxML/Zygote.jl))
- [TensorSketch](https://github.com/kjslag/fastDisentangle/blob/main/fastDisentangle.jl)
- [Documenter.jl](https://github.com/juliadocs/documenter.jl) (for documentation)
- [Runic.jl](https://github.com/fredrikekre/Runic.jl) (for code formatting)

The disentangling algorithms are designed to return `ITensor` data structures to better integrate into the Julia ecosystem, however the internals are completely pure Julia multi-dimensional arrays.

## Examples

We employ the various disentangler algorithms to disentangle various tensor network representations:
- Matrix Product State (MPS) representations, see [Disentangling Matrix Product States](@ref DisentanglingMPS)
- Matrix Product Operator (MPO) representations, see [Unitary Compression](@ref DisentanglingMPO)
- Locally Purified Density Operator (LPDO) representations, see [Optimizing the representation of LPDO](@ref DisentanglingLPDO)

These represent a sample and can be extended to other scenarios, or other representations that allow isometric representational optimization.

## Profiling the performance of disentanglers

Functions to profile the different disentangling algorithms are outlined in the [Profiling disentanglers](@ref DisentanglingPerf) section in the documentation.

## Enabling GPU compute

[GPU integration](@ref gpu_integration) section in the documentation outlines a procedure that enables computing on GPUs.

## AI usage

AI agents weren't put to use to generate any code, this is in order to retain and experience the process of learning, writing code and building software packages! However, AI was used to generate docstrings, the authors have verified the generated docstrings and have mostly modified them!

## Authors

- Amit Jamadagni (gangapurama@ornl.gov)
- Eugene Dumitrescu (dumitrescuef@ornl.gov)

## Acknowledgment

We acknowledge the support by the U.S. Department of Energy, Office of Science, Advanced Scientific Research Program, Early Career Award under contract number ERKJ420. We thank Bo Xiao, Nathan Johnson, George Arapantonis, Varadharajan Muruganandam for insightful discussions.
