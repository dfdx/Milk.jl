
"""
Cross-entropy loss as defined in PyTorch:

http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss

This method works with one-hot-encoded inputs
"""
function pytorch_crossentropy(ŷ::AbstractMatrix, y::AbstractMatrix)
    sublosses = -sum(y .* ŷ, 1) .+ log.(sum(exp.(ŷ), 1))
    return mean(sublosses)
end


# from Flux
function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
    return -sum(y .* log.(ŷ) .* weight) / size(y, 2)
end


"""
Negative log-likelihood. ŷ should be a vector of log probabilities.
"""
nll(ŷ::AbstractVector, c::Real) = -ŷ[c]

function nll(ŷ::AbstractMatrix, c::Vector{<:Real})
    loss = 0
    for j=1:size(ŷ, 2)
        i = c[j]
        loss += -ŷ[i, j]
    end
    return loss
end


nll_grad(ŷ::AbstractVector, c::Int) = ŷ[c] - 1

function nll_grad(ŷ::AbstractMatrix, c::Vector{Int})
    # TODO: actually we are looking for dŷ::AbstractMatrix
    dx = typeof(ŷ)(undef, length(c))
    for j=1:size(ŷ, 2)
        i = c[j]
        dx[j] = exp(ŷ[i, j] - 1)
    end
    return dx
end


## negative log likelihood for proper probability vector/matrix ŷ
## NOTE: just a scratch, not tested, API not confirmed


# TODO: apply log to nll and nll_grad params (ŷ - logprobability)
# TODO: @test nll(softmax(x)) == crossentropy(x)
# nll(ŷ::AbstractVector, c::Int) = -log(ŷ[c])

# function nll(ŷ::AbstractMatrix, c::Vector{Int})
#     loss = 0
#     for j=1:size(ŷ, 2)
#         i = c[j]
#         loss += -log(ŷ[i, j])
#     end
#     return loss
# end


# nll_grad(ŷ::AbstractVector, c::Int) = ŷ[c] - 1

# function nll_grad(ŷ::AbstractMatrix, c::Vector{Int})
#     dx = typeof(ŷ)(undef, length(c))
#     for j=1:size(ŷ, 2)
#         i = c[j]
#         dx[j] = ŷ[i, j] - 1
#     end
#     return dx
# end



@require CuArrays begin

    # have to split sublosses into 2 subexpressions since otherwise CUDAnative failes with
    # 'ERROR: Broadcast output type Any is not concrete'
    function cross_entropy_loss(ŷ::CuArray{T,2}, y::CuArray{T,2}) where T        
        w1 = -sum(y .* ŷ, 1)
        w2 = sum(exp.(ŷ), 1)
        sublosses = w1 .+ CUDAnative.log.(w2)
        return mean(sublosses)
    end

end
