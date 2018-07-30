## mlutils.jl - a set of ML utilities for parameter initalization, performance metrics, etc.

## initialization

function xavier(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(dim_in, dim_out) .* (high - low) .+ low
end

## one-hot encoding

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


## metrics

function accuracy(ŷ::AbstractMatrix, y::AbstractMatrix)
    return mean(findmax(@view ŷ[:,i])[2] == findmax(@view y[:,i])[2] for i=1:size(ŷ,2))
end



## eachbatch

struct EachBatch{T}
    src::AbstractArray{T,2}
    batch_size::Int
end


function Base.iterate(itr::EachBatch, start::Int)
    if start > size(itr.src, 2)
        return nothing
    else
        val = itr.src[:, start : min(start + itr.batch_size - 1, size(itr.src, 2))]
        next_state = start + itr.batch_size
        return val, next_state
    end
end
Base.iterate(itr::EachBatch) = iterate(itr, 1)


function eachbatch(X::AbstractArray{T,2}; size=100) where T
    return EachBatch(X, size)
end


## CUDA support

@require CuArrays begin

    """
    Convert to a corresponding CUDA object.

    For arrays returns an instance of CuArray.
    For structs returns similar struct with all arrays fields converted to CuArray.
    """
    to_cuda(a::AbstractArray{T,N}) where {T,N} = convert(CuArray{Float32,N}, a)
    to_cuda(a::CuArray) = a
    
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

end
