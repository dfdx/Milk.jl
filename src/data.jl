using Printf
using Statistics
using Random


function _read_mnist_csv(path, count)
    path = expanduser(path)
    X = Array{UInt8}(undef, 784, count)
    y = Array{Int}(undef, 60_000)
    lines = open(readlines, path)
    for i=1:count
        nums = [parse(UInt8, x) for x in split(lines[i], ",")]
        y[i] = nums[1]
        X[:, i] = nums[2:end]
        if i % 5000 == 0
            Printf.@printf("Loaded %3.2f%%\n", i / count * 100)
        end
    end
    y = [n == 0 ? 10 : n for n in y]  # replace 0 with 10 so numbers can be used as indices 
    return X ./ 256, y
end


function read_mnist()
    xtrn, ytrn = _read_mnist_csv("~/data/common/mnist_train.csv", 60_000)
    xtst, ytst = _read_mnist_csv("~/data/common/mnist_test.csv", 10_000)
    return xtrn, ytrn, xtst, ytst
end


# this is a slightly modified version of housing.jl from Knet.jl repository
"""
    housing([test]; [url, file])
Return (xtrn,ytrn,xtst,ytst) from the [UCI Boston
Housing](https://archive.ics.uci.edu/ml/machine-learning-databases/housing)
dataset The dataset has housing related information for 506
neighborhoods in Boston from 1978. Each neighborhood has 14
attributes, the goal is to use the first 13, such as average number of
rooms per house, or distance to employment centers, to predict the
14’th attribute: median dollar value of the houses.
`test=0` by default and `xtrn` (13,506) and `ytrn` (1,506) contain the
whole dataset. Otherwise data is shuffled and split into train and
test portions using the ratio given in `test`. xtrn and xtst are
always normalized by subtracting the mean and dividing into standard
deviation.
"""
function housing(test=0.0;
                 file=expanduser("~/data/common/housing.data"),
                 url="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data")
    if !isfile(file)
        isdir(dirname(file)) || mkpath(dirname(file))
        info("Downloading $url to $file")
        download(url, file)
    end
    data = readdlm(file)'
    # @show size(data) # (14,506)
    x = data[1:13,:]
    y = data[14:14,:]
    x = (x .- mean(x,2)) ./ std(x,2) # Data normalization
    if test == 0
        xtrn = xtst = x
        ytrn = ytst = y
    else
        r = randperm(size(x,2))          # trn/tst split
        n = round(Int, (1-test) * size(x,2))
        xtrn=x[:,r[1:n]]
        ytrn=y[:,r[1:n]]
        xtst=x[:,r[n+1:end]]
        ytst=y[:,r[n+1:end]]
    end
    return (xtrn, ytrn, xtst, ytst)
end
