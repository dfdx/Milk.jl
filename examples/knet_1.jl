include("../src/core.jl")

mutable struct Linear{T}
    w::AbstractMatrix{T}
    b::Real
end

predict(m::Linear,x) = m.w*x .+ m.b

loss(m,x,y) = mean(abs2.(y - predict(m, x)))

function train(m::Linear, data; lr=.1)
    for (x,y) in data
        loss_val, g = grad(loss, m, x, y)
        update!(m, g[1], (x, gx) -> x .- lr .* gx)
    end
    return m
end


function main()
    x,y = housing()
    m = Linear(0.1*randn(1,13), 0.0)
    for i=1:10
        train(m, [(x,y)])
        println(loss(m,x,y))
    end
end
