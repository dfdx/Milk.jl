include("../src/core.jl")

using Yota
using GradDescent
using Statistics
import Printf



# variational autoencoder with Gaussian observed and latent variables
mutable struct VAE{T}
    # encoder / recognizer
    We1::AbstractMatrix{T}  # encoder: layer 1 weight
    be1::AbstractVector{T}  # encoder: layer 1 bias
    We2::AbstractMatrix{T}  # encoder: layer 2 weight
    be2::AbstractVector{T}  # encoder: layer 2 bias
    We3::AbstractMatrix{T}  # encoder: layer 3 mu weight
    be3::AbstractVector{T}  # encoder: layer 3 mu bias
    We4::AbstractMatrix{T}  # encoder: layer 3 log(sigma^2) weight
    be4::AbstractVector{T}  # encoder: layer 3 log(sigma^2) bias
    # decoder / generator
    Wd1::AbstractMatrix{T}  # decoder: layer 1 weight
    bd1::AbstractVector{T}  # decoder: layer 1 bias
    Wd2::AbstractMatrix{T}  # decoder: layer 2 weight
    bd2::AbstractVector{T}  # decoder: layer 2 bias
    Wd3::AbstractMatrix{T}  # decoder: layer 3 weight
    bd3::AbstractVector{T}  # decoder: layer 3 bias
end

function Base.show(io::IO, m::VAE{T}) where T
    print(io, "VAE{$T}($(size(m.We1,2)), $(size(m.We1,1)), $(size(m.We2,1)), " *
          "$(size(m.We3,1)), $(size(m.Wd1,1)), $(size(m.Wd2,1)), $(size(m.Wd3,1)))")
end


VAE{T}(n_inp, n_he1, n_he2, n_z, n_hd1, n_hd2, n_out) where T =
    VAE{T}(
        # encoder
        xavier(n_he1, n_inp),
        zeros(n_he1),
        xavier(n_he2, n_he1),
        zeros(n_he2),
        xavier(n_z, n_he2),
        zeros(n_z),
        xavier(n_z, n_he2),
        zeros(n_z),
        # decoder
        xavier(n_hd1, n_z),
        zeros(n_hd1),
        xavier(n_hd2, n_hd1),
        zeros(n_hd2),
        xavier(n_out, n_hd2),
        zeros(n_out)
    )


function encode(m::VAE, x)
    he1 = softplus.(m.We1 * x .+ m.be1)
    he2 = softplus.(m.We2 * he1 .+ m.be2)
    mu = m.We3 * he2 .+ m.be3
    log_sigma2 = m.We4 * he2 .+ m.be4
    return mu, log_sigma2
end

function decode(m::VAE, z)
    hd1 = softplus.(m.Wd1 * z .+ m.bd1)
    hd2 = softplus.(m.Wd2 * hd1 .+ m.bd2)
    x_rec = logistic.(m.Wd3 * hd2 .+ m.bd3)
    return x_rec
end


function loss_fn(m::VAE, eps, x)
    mu, log_sigma2 = encode(m, x)
    z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
    x_rec = decode(m, z)
    # loss
    rec_loss = -sum(x .* log.(1e-10 .+ x_rec) .+ (1 .- x) .* log.(1e-10 + 1 .- x_rec), dims=1)
    KLD = -0.5 * sum(1 .+ log_sigma2 .- mu .^ 2.0 .- exp.(log_sigma2), dims=1)
    cost = mean(rec_loss .+ KLD)
end


function fit!(m::VAE{T}, X::AbstractMatrix{T};
              n_epochs=10, batch_size=100, opt=Adam(α=0.001), cuda=false) where T
    for epoch in 1:n_epochs
        print("Epoch $epoch: ")
        epoch_loss = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            eps = typeof(x)(randn(size(m.We3, 1), batch_size))
            loss, g = grad(loss_fn, m, eps, x)
            update!(m, g[1], (x, gx) -> (x .- 0.001 .* gx))
            epoch_loss += loss
            # println("batch_loss = $loss")
            # println("sum(m.We1) = $(sum(m.We1))")
        end
        println("avg_loss=$(epoch_loss / (size(X,2) / batch_size)), elapsed=$t")
    end
    return m
end


function generate(m::VAE, z::AbstractVector)
    return decode(m, z)
end


function generate(m::VAE, id::Int)
    z = zeros(size(m.Wd1, 2))
    z[id] = 1.0
    hd1 = tanh.(m.Wd1 * z .+ m.bd1)
    hd2 = tanh.(m.Wd2 * hd1 .+ m.bd2)
    return logistic.(m.Wd3 * hd2 .+ m.bd3)
end


function reconstruct(m::VAE, x::AbstractVector)
    x = reshape(x, length(x), 1)
    mu, _ = encode(m, x)
    z = mu
    x_rec = decode(m, z)
    return x_rec
end



function main()
    X, _ = read_mnist()
    m = VAE{Float64}(784, 500, 200, 20, 200, 500, 784)
    fit!(m, X)
    
    sum(abs2, x .- reconstruct(m, x)) |> println
end
