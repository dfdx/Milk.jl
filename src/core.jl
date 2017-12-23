using Espresso
using XGrad
using GradDescent
using Distributions
using NNlib
using Requires


# otherwise CuArray type isn't resolved during loading
@require CuArrays using CuArrays

include("utils.jl")
include("diffrules.jl")
include("inplacerules.jl")
include("losses.jl")
include("modelopt.jl")
include("mlutils.jl")
