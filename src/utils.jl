
function dive_in(mod)
    for n in names(mod, true)
        @eval import $(Symbol(mod)): $n
    end
end
