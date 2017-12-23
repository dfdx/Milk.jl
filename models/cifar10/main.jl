
using CuArrays
using CUDNN
# using Milk
include("../../src/core.jl")

using MLDatasets
using MLDataUtils
using PyPlot

include("model.jl")

function main()
    images, labels = CIFAR10.traindata()
    X_train = images
    # X_train = normalize(X_train)
    y_train = one_hot(maximum(labels) + 1, labels .+ 1)
    m = CIFAR10Model()
    @time m = fit!(m, X_train, y_train; cuda=true, n_epochs=100)

    # sample accuracy
    X = X_train[:,:,:,1:100] |> to_cuda
    y = y_train[:,1:100] |> to_cuda
    accuracy(predict(m, X), y)
end



# function normalize(X)
#     μ = mean(X, (1,2))
#     # σ2 = std(X, (1,2))
#     σ2 = 0.5
#     return (X .- μ) ./ σ2
# end
