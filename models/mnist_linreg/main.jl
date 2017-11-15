
using CuArrays
include("../../src/core.jl")

using MLDataUtils
using MLDatasets

using CuConv

include("model.jl")


@diffrule CUDAnative.log(x::Real) x ds / x
@diffrule CUDAnative.log(x::AbstractArray) x ds ./ x

function main()
    data, labels = MNIST.traindata()
    X_train = reshape(data, 28, 28, 1, size(data, 3))
    y_train = one_hot(maximum(labels) + 1, labels .+ 1)
    m = C2(randn(5, 5, 1, 3), randn(5, 5, 3, 6), randn(10, 4*4*6), randn(10))
    m = fit!(m, X_train, y_train; cuda=true, n_epochs=700)
end



function main_1420()
    data, labels = MNIST.traindata()
    X_train = reshape(data, 28, 28, 1, size(data, 3))
    y_train = one_hot(maximum(labels) + 1, labels .+ 1)
    X = to_cuda(X_train[:,:,:,101:200])
    y = to_cuda(y_train[:,101:200])   
    # m = C2(randn(5, 5, 1, 3), randn(10, 4*4*6), randn(10)) |> to_cuda
    inputs = [:m => m, :X => X, :y => y]
    ctx = Dict()
    f = loss
    xdiff(f; ctx=ctx, inputs...)


    ŷ = predict(m, X)
end


# fix for `log.(::CuArray)` - after `CUDAnative.log` has been called at least ones,
# `log` start working with CuArray too



# __a = rand(5)
# CUDAnative.log.(__a)
