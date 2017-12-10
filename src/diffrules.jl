
# @diffrule conv2d(x, w) w conv2d_grad_w(x, w, ds)
# @diffrule conv2d(x, w) x conv2d_grad_x(x, w, ds)

# @diffrule conv2d(x, w; _opts...) w conv2d_grad_w(x, w, ds; _opts...)
# @diffrule conv2d(x, w; _opts...) x conv2d_grad_x(x, w, ds; _opts...)

# @diffrule pool(x) x pool_grad(x, pool(x), ds)
# @diffrule pool(x; _opts...) x pool_grad(x, pool(x), ds; _opts...)

@diffrule NNlib.conv2d(x, w) w NNlib.conv2d_grad_w(x, w, ds;)
@diffrule NNlib.conv2d(x, w) x NNlib.conv2d_grad_x(x, w, ds)

@diffrule NNlib.conv2d(x, w; _opts...) w NNlib.conv2d_grad_w(x, w, ds; _opts...)
@diffrule NNlib.conv2d(x, w; _opts...) x NNlib.conv2d_grad_x(x, w, ds; _opts...)

@diffrule NNlib.pool(x) x NNlib.pool_grad(x, pool(x), ds)
@diffrule NNlib.pool(x; _opts...) x NNlib.pool_grad(x, pool(x), ds; _opts...)

@diffrule NNlib.σ(x::Number) x (NNlib.σ(x) .* (1 .- NNlib.σ(x)) .* ds)
@diffrule NNlib.sigmoid(x::Number) x (NNlib.σ(x) .* (1 .- NNlib.σ(x)) .* ds)

@diffrule NNlib.relu(x::Number) x (x .> 0) .* ds

# logistic(x) = 1 ./ (1 + exp.(-x))
# @diffrule logistic(x::Number) x (logistic(x) .* (1 .- logistic(x)) .* ds)

# softplus(x) = log(exp(x) + 1)
# @diffrule softplus(x::Number) x logistic(x) .* ds



@diffrule CUDAnative.log(x::Real) x ds / x
@diffrule CUDAnative.log(x::AbstractArray) x ds ./ x
