function sig = sigmoid(x)
    sig = exp(x) ./ (1 + exp(x));
end