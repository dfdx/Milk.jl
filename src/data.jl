using Printf

function read_mnist()
    path = expanduser("~/data/common/mnist_train.csv")
    X = Array{UInt8}(undef, 784, 60_000)
    y = Array{Int}(undef, 60_000)
    lines = open(readlines, path)
    for i=1:60_000
        nums = [parse(UInt8, x) for x in split(lines[i], ",")]
        y[i] = nums[1]
        X[:, i] = nums[2:end]
        if i % 5000 == 0
            Printf.@printf("Loaded %3.2f%%\n", i / 60_000 * 100)
        end
    end
    return X ./ 256, y
end
