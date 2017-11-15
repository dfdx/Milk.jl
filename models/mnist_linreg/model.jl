

mutable struct C2
    Wc1::AbstractArray{T,4} where T
    Wc2::AbstractArray{T,4} where T
    W1::AbstractMatrix
    b::AbstractVector
end

C2() = C2(rand(1,1,1,1), rand(1,1,1,1), rand(1,1), rand(1))



function predict(m, X::AbstractArray)
    c1 = conv2d(X, m.Wc1)
    p1 = pool(c1)
    c2 = conv2d(p1, m.Wc2)
    p2 = pool(c2)
    cc = reshape(p2, 4*4*6, size(p1, 4))
    res = logistic.(m.W1 * cc .+ m.b)
    return res
end


function loss(m, X::AbstractArray, y::AbstractMatrix)
    ŷ = predict(m, X)
    # cost = sum((y .- ŷ) .^ 2)
    cost = pytorch_cross_entropy(ŷ, y)
    return cost
end


function partial_fit!(m, X::AbstractArray, y::AbstractMatrix; mem=Dict())
    # opt = ModelOptimizer(m, Momentum(η=0.001, γ=0.9))
    opt = ModelOptimizer(m, Adam())
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
        for (x, y) in batchview((X, Y))
            x, y = map(copy, (x, y))
            if cuda
                x, y = map(to_cuda, (x, y))
            end
            cost = partial_fit!(m, x, y)            
        end
        println(cost)
    end
    return m
end


function predict_label(m, X)
    findmax(predict(m, X))[2]
end
