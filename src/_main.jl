include("core.jl")

import Yota: grad!, TArray, TReal, tracked, Tape

nll(ŷ::AbstractVector, c::Int) = -ŷ[c]
nll_grad(ŷ::AbstractVector, c::Int) = ŷ[c] - 1


nll(ŷ::TArray{T,1}, c::Int) where T = record!()

function main_14()
    
end
