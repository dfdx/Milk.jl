

function cross_entropy(y::AbstractMatrix, ŷ::AbstractMatrix)
    return -mean(sum(y .* log.(ŷ), 1))
end

# TODO: add cross_entropy derivative (generate and manually optimize)

