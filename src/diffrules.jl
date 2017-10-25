
@diffrule conv2d(x, w; _opts...) w conv2d_grad_w(x, w, ds; _opts...)
@diffrule conv2d(x, w; _opts...) x conv2d_grad_x(x, w, ds; _opts...)

@diffrule pool(x; _opts...) x pool_grad(x, y, ds; _opts...)

logistic(x) = 1 ./ (1 + exp.(-x))
@diffrule logistic(x::Number) x (logistic(x) .* (1 .- logistic(x)) .* ds)

softplus(x) = log(exp(x) + 1)
@diffrule softplus(x::Number) x logistic(x) .* ds
