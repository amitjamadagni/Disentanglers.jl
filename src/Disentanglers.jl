module Disentanglers

"""
Module Disentanglers provides a suite of solvers that aim to efficiently
disentangle a tensor network.
"""

# ----------------------- Dependencies -----------------------

using ITensors
using ITensorMPS
using Zygote
using ChainRules
using ChainRulesCore
using Manifolds
using Manopt
using Statistics
using DifferentiationInterface
using LinearAlgebra
using Accessors
using CUDA
using Adapt

# --------------- Include files with exports in them  ---------------

include("utils/utils.jl")

# ------------ Include different isometric routines  -------------

include("opt/ro_are_iso.jl")
include("opt/ro_sre_iso.jl")
include("opt/ro_overlap_iso.jl")
include("opt/ro_nfn_iso.jl")

include("non_opt/ev_iso.jl")
include("non_opt/ts_module.jl")
include("non_opt/ts_iso.jl")

include("../examples/mps_prep_ex.jl")
include("../examples/mpo_compression_disentangler_ex.jl")
include("../examples/mpo_compression_distance_measure_ex.jl")
include("../examples/lpdo_opt_ex.jl")

include("../benchmarks/profiling_disentanglers.jl")
end
