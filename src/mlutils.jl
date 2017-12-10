
function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end


function class_to_index(classes)
    classes = sort(classes)
    return Dict(c => i for (i, c) in enumerate(classes))
end


function one_hot(::Type{T}, num_classes::Int, vals::AbstractVector{<:Integer}) where T
    ret = zeros(T, num_classes, length(vals))
    for (i, v) in enumerate(vals)
        ret[v, i] = T(1)
    end
    return ret
end

one_hot(num_classes::Integer, vals) = one_hot(Float64, num_classes, vals)

function one_hot(::Type{T}, c2i::Dict, vals) where T
    result = zeros(T, length(c2i), length(vals))
    for (j, val) in enumerate(vals)
        i = c2i[val]
        result[i, j] = one(T)
    end
    return result
end

one_hot(c2i::Dict, vals) = one_hot(Float64, c2i, vals)


function accuracy(ŷ::AbstractMatrix, y::AbstractMatrix)
    return mean(findmax(@view ŷ[:,i])[2] == findmax(@view y[:,i])[2] for i=1:size(ŷ,2))
end




# todo: don't depend on CuArrays
using CuArrays

to_cuda(a::AbstractArray{T,N}) where {T,N} = convert(CuArray{Float32,N}, a)
to_cuda(a::CuArray) = a

import Espresso

function to_cuda(m)
    if isstruct(m)
        new_m = Espresso.struct_like(m)
        for fld in fieldnames(typeof(m))
            v = getfield(m, fld)
            new_v = to_cuda(v)
            setfield!(new_m, fld, new_v)
        end
        return new_m
    else
        return m
    end
end



