# using CuArrays
include("../../src/core.jl")

# using MLDataUtils
using MLDatasets


include("model.jl")


@diffrule CUDAnative.log(x::Real) x ds / x
@diffrule CUDAnative.log(x::AbstractArray) x ds ./ x

function main()
    data, labels = MNIST.traindata()
    X_train = reshape(data, 28, 28, 1, size(data, 3))
    y_train = one_hot(maximum(labels) + 1, labels .+ 1)
    m = C2(randn(5, 5, 1, 3), randn(5, 5, 3, 6), randn(10, 4*4*6), randn(10))
    m = fit!(m, X_train, y_train; cuda=true, n_epochs=700)
    # test accuracy on a single batch
    X = X_train[:,:,:,101:200]
    y = y_train[:,101:200]
    accuracy(predict(m, X), y)
end
