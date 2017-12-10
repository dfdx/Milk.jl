module Milk

using Reexport

@reexport using XGrad
@reexport using NNlib
@reexport using GradDescent

export
    # utils
    xavier_init,
    class_to_index,
    one_hot,    
    # gradient descent
    ModelOptimizer,
    update_params!,
    # losses
    cross_entropy_loss,
    # other
    accuracy,
    to_cuda

include("core.jl")

end # module
