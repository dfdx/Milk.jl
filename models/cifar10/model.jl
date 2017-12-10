
mutable struct CIFAR10Model
    # W* - weight; b* - bias; c1 - convolution 1; fc1 - fully connected 1
    Wc1    # (H,W,I,O) == (5,5,3,6)
    Wc2    # (H,W,I,O) == (5,5,6,16)
    Wfc1   # (120, 16*5*5)
    bfc1   # (120,)
    Wfc2   # (84,120)
    bfc2   # (84,)
    Wfc3   # (10,84)
    bfc3   # (10,)
end

CIFAR10Model() = CIFAR10Model(
    randn(5, 5, 3, 6) / (5 * 5 * 3 * 6),
    randn(5, 5, 6, 16) / (5 * 5 * 6 * 16),
    xavier_init(120, 16*5*5),
    randn(120) / 120,
    xavier_init(84, 120),
    randn(84) / 84,
    xavier_init(10, 84),
    randn(10) / 10
)

# mutable struct CIFAR10Model
#     # W* - weight; b* - bias; c1 - convolution 1; fc1 - fully connected 1
#     # Wc1    # (H,W,I,O) == (5,5,3,6)
#     Wfc3   # (10,84)
#     bfc3   # (10,)
# end

# CIFAR10Model() = CIFAR10Model(
#     xavier_init(10, 32 * 32 * 3),
#     randn(10)
# )


Base.show(io::IO, m::CIFAR10Model) = print(io, "CIFAR10Model()")


function predict(m::CIFAR10Model, X::AbstractArray{<:AbstractFloat, 4})
    c1 = conv2d(X, m.Wc1)  # 32x32x3xN => 28x28x6xN
    p1 = pool(relu.(c1); stride=2)
    c2 = conv2d(p1, m.Wc2)  # 14x14x6xN => 10x10x16xN
    p2 = pool(relu.(c2); stride=2)
    c2_f = reshape(p2, 400, size(p2, 4))
    fc1 = relu.(m.Wfc1 * c2_f .+ m.bfc1)
    fc2 = relu.(m.Wfc2 * fc1 .+ m.bfc2)
    fc3 = sigmoid.(m.Wfc3 * fc2 .+ m.bfc3)
    # fc3 = m.Wfc3 * fc2 .+ m.bfc3
    return fc3
end


# function predict(m::CIFAR10Model, X::AbstractArray{<:AbstractFloat, 4})
#     c1_f = reshape(X, 32 * 32 * 3, size(X, 4))
#     # c1 = conv2d(X, m.Wc1)  # 32x32x3xN => 28x28x6xN
#     # c1_f = reshape(c1, 4704, size(c1, 4))
#     fc3 = sigmoid.(m.Wfc3 * c1_f .+ m.bfc3)
#     return fc3
# end


function loss(m::CIFAR10Model, X::AbstractArray{F, 4}, y::AbstractMatrix{F}) where {F <:AbstractFloat}
    ŷ = predict(m, X)
    return pytorch_cross_entropy(ŷ, y)
end

# function loss(m::CIFAR10Model, X::AbstractArray{F, 4}, y::AbstractMatrix{F}) where {F <:AbstractFloat}
#     ŷ = predict(m, X)
#     return cross_entropy(y, ŷ)
# end


function partial_fit!(m::CIFAR10Model, X::AbstractArray, y::AbstractMatrix; mem=Dict())
    opt = ModelOptimizer(m, Momentum(η=0.05, γ=0.9))
    # opt = ModelOptimizer(m, Adam(α=0.01))
    cost, dm, dX, dy = xgrad(loss; mem=mem, m=m, X=X, y=y)
    update_params!(opt, m, dm)
    return cost
end


function fit!(m, X::AbstractArray, Y::AbstractMatrix; cuda=false, n_epochs=100)
    mem = Dict()
    if cuda
        m = to_cuda(m)
    end
    @time for epoch=1:n_epochs
        print("Epoch: $epoch: ")
        cost = 0
        for (x, y) in batchview((X, Y); size=100)
            x, y = map(copy, (x, y))
            if cuda
                x, y = map(to_cuda, (x, y))
            end
            cost = partial_fit!(m, x, y; mem=mem)
        end
        println(cost)
    end
    return m
end


# function fit!(m::CIFAR10Model, X::AbstractArray, Y::AbstractMatrix)
#     mem = Dict()
#     for epoch=1:10
#         info("Epoch: $epoch")
#         for (x, y) in batchview((X, Y))
#             cost = partial_fit!(m, copy(x), copy(y); mem=mem)
#             info("cost: $cost")
#         end
#     end
#     return m
# end
