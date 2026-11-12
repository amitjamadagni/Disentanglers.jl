using Documenter
using Disentanglers

DocMeta.setdocmeta!(Disentanglers, :DocTestSetup, :(using Disentanglers); recursive = true)

# checked_modules = [Disentanglers]

# for sym in names(Disentanglers, all=true)
#     if isdefined(Disentanglers, sym)
#         mod = getfield(Disentanglers, sym)
#         if mod isa Module && mod !== Disentanglers.TensorSketch && mod !== Disentanglers
#             push!(checked_modules, mod)
#         end
#     end
# end

# println(checked_modules)


makedocs(;

    # use this for generating online docs!
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),

    # format = Documenter.LaTeX(show_log = true),

    modules = [Disentanglers],
    checkdocs_ignored_modules = [Disentanglers.TensorSketch],
    sitename = "Disentanglers.jl",
    repo = Documenter.Remotes.GitHub("amitjamadagni", "Disentanglers.jl"),
    # remotes = nothing,
    checkdocs = :exports,
    pages = [
        "Introduction" => "index.md",

        "Disentangling Algorithms" => [
            "Riemannian solvers" => "algos/reimannian_solvers.md",
            "Tensor sketching" => "algos/tensor_sketch.md",
            "Evenbly-Vidal" => "algos/evenbly_vidal.md",
        ],

        "Examples" => [
            "State preparation" => "ex/state_preparation.md",
            "Unitary compression" => "ex/unitary_compression.md",
            "LPDO optimization" => "ex/lpdo_optimization.md",
        ],

        "Profiling disentanglers" => "profiling/profiling.md",

        "API Reference" => [
            "Optimization functions" => "api/optimization_functions.md",
            "Utility functions" => "api/utilities.md",
        ],

        "Future Directions" => [
            "Integrating new objective functions" => "fd/integrating_new_obj_funcs.md",
            "GPU Integration" => "fd/gpu_integration.md",
        ],
    ]
)
