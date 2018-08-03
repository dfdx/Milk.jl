include("../src/core.jl")

import NNlib: logsoftmax

mutable struct Linear{T}
    w::AbstractMatrix{T}
    b::AbstractMatrix{T}
end

predict(m,x) = logsoftmax(m.w*x .+ m.b)

loss(m,x,ygold) = nll(predict(m,x), ygold)


function train(m::Linear, data; lr=0.1)
    for (x,y) in data
        x = convert(Array{Float32}, x); y = copy(y)
        loss_val, g = grad(loss, m, x, y)
        update!(m, g[1], (x, gx) -> x .- Float32(lr) .* gx)
        println(loss_val)
    end
    return m
end


function main()
    xtrn, ytrn, xtst, ytst = read_mnist()
    dtrn = zip(eachbatch(xtrn), eachbatch(ytrn))
    dtst = zip(eachbatch(xtst), eachbatch(ytst))
    m = Linear(0.1f0*randn(Float32,10,784), zeros(Float32,10,1))
    # println((:epoch, 0, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
    for epoch=1:10
        train(m, dtrn; lr=0.5)
        # println((:epoch, epoch, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
    end
end
