
import Yota: grad!, Call, TAny, TArray, TReal, record!
import NNlib: logsoftmax, ∇logsoftmax

# logistic
logistic(x) = 1 / (1 + exp(-x))
logistic(x::TReal) = record!(x.tape, Call, logistic, (x,))
function grad!(dy::TAny, ::Val{1}, op::Call{typeof(logistic), Tuple{TReal}})
    x = op.args[1]
    ll = logistic(x)
    return ll * (1 - ll) * dy
end

# softplus
softplus(x) = log(exp(x) + 1)
softplus(x::TReal) = record!(x.tape, Call, softplus, (x,))
function grad!(dy::TAny, ::Val{1}, op::Call{typeof(softplus), Tuple{TReal}})
    x = op.args[1]
    return logistic(x) * dy
end


## NNlib
NNlib.logsoftmax(x::TArray) = record!(x.tape, Call, logsoftmax, (x,))
grad!(dy::TAny, ::Val{1}, op::Call{typeof(logsoftmax), Tuple{TArray}}) =
    record!(x.tape, Call, ∇logsoftmax, (dy, op.args[1]))
