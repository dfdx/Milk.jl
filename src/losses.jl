
# TODO: separate cross_entropy for CuArray and CUDAnative.log

# function cross_entropy(ŷ::AbstractMatrix, y::AbstractMatrix)
#     # although it's correct definition of cross-entropy, it's not divided by
#     # ŷ[i != class]; this way optimizer tends to make all elements in ŷ to be 1.0 
#     return -mean(sum(y .* log.(ŷ), 1))
# end

# @require CuArrays begin
    
#     function cross_entropy(ŷ::CuArray{T,2}, y::CuArray{T,2}) where T
#         return -mean(sum(y .* CUDAnative.log.(ŷ), 1))
#     end
#     # TODO: add cross_entropy derivative (generate and manually optimize)
    
# end

"""
Cross-entropy loss as defined in PyTorch:

http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss

This method works with one-hot-encoded inputs
"""
function cross_entropy_loss(ŷ::AbstractMatrix, y::AbstractMatrix)
    sublosses = -sum(y .* ŷ, 1) .+ log.(sum(exp.(ŷ), 1))
    return mean(sublosses)
end


function cross_entropy_loss(ŷ::CuArray{T,2}, y::CuArray{T,2}) where T
    sublosses = -sum(y .* ŷ, 1) .+ CUDAnative.log.(sum(exp.(ŷ), 1))
    return mean(sublosses)
end



# function cross_entropy_loss(ŷ::AbstractMatrix, y::AbstractVector{<:Integer})
#     # TODO: convert to one-hot and use pytroch_cross_entropy
# end
