
"""
Cross-entropy loss as defined in PyTorch:

http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss

This method works with one-hot-encoded inputs
"""
function cross_entropy_loss(ŷ::AbstractMatrix, y::AbstractMatrix)
    sublosses = -sum(y .* ŷ, 1) .+ log.(sum(exp.(ŷ), 1))
    return mean(sublosses)
end


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
