
using Espresso
using XGrad
using GradDescent
using Distributions
using NNlib
# using Requires
using CuArrays  # TODO: make soft dependency instead

# include("nnlib.jl")
include("utils.jl")
include("diffrules.jl")
include("inplacerules.jl")
include("losses.jl")
include("modelopt.jl")
include("mlutils.jl")

