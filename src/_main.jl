include("core.jl")


function main_14()
    val, g = grad(nll, logsoftmax(rand(4, 3)), [1, 2, 1])
end
