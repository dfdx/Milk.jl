
using CuArrays
using CuConv
include("../../src/core.jl")

using MLDatasets
using MLDataUtils
using PyPlot


include("model.jl")


@diffrule CUDAnative.log(x::Real) x ds / x
@diffrule CUDAnative.log(x::AbstractArray) x ds ./ x


function main()
    images, labels = CIFAR10.traindata()
    X_train = images
    y_train = one_hot(maximum(labels) + 1, labels .+ 1)
    m = CIFAR10Model()
    m = fit!(m, X_train, y_train; cuda=true, n_epochs=100)

    X = X_train[:,:,:,1:100] |> to_cuda
    y = y_train[:,1:100] |> to_cuda
end
