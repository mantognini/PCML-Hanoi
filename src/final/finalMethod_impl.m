
function [kyTrPred, kyVaPred, kyTePred] = finalMethod_impl(kXTr, kyTr, kXVa, kXTe, k)
    D = size(kXTr, 2);
    kPhis = buildPhis(D, k);
    ktXTr = mapPhis(kPhis, kXTr);
    ktXVa = mapPhis(kPhis, kXVa);
    ktXTe = mapPhis(kPhis, kXTe);

    K = 10;
    kLambda = bestLambdaKFold(kyTr, ktXTr, K);
    kBeta = ridgeRegression(kyTr, ktXTr, kLambda);

    kyTrPred = ktXTr * kBeta;
    kyVaPred = ktXVa * kBeta;
    kyTePred = ktXTe * kBeta;
end
      
function phis = buildPhis(D, k)
    % Build basis functions
    power = @(i, x) x .^ i;
    %sigmoid = @(x) exp(x) / (1 + exp(x));

    p = 1;
    phis{p} = @(x) 1;

    for d = 1:D
        if k == 1
%             p = p + 1; phis{p} = @(x) sign(x(d)) * power(1/2, sign(x(d)) * x(d));
            p = p + 1; phis{p} = @(x) power(1, x(d));
            p = p + 1; phis{p} = @(x) power(2, x(d));
            p = p + 1; phis{p} = @(x) power(3, x(d));
%             p = p + 1; phis{p} = @(x) power(4, x(d));
%             p = p + 1; phis{p} = @(x) power(5, x(d));
%             p = p + 1; phis{p} = @(x) sigmoid(x(d));
        end

        if k == 2
%             p = p + 1; phis{p} = @(x) sign(x(d)) * power(1/2, sign(x(d)) * x(d));
%             p = p + 1; phis{p} = @(x) power(1, x(d));
            p = p + 1; phis{p} = @(x) power(2, x(d));
            p = p + 1; phis{p} = @(x) power(3, x(d));
%             p = p + 1; phis{p} = @(x) power(4, x(d));
%             p = p + 1; phis{p} = @(x) power(5, x(d));
%             p = p + 1; phis{p} = @(x) power(6, x(d));
%             p = p + 1; phis{p} = @(x) sigmoid(x(d));
%             p = p + 1; phis{p} = @(x) tanh(x(d));
        end

        if k == 3
%             p = p + 1; phis{p} = @(x) sign(x(d)) * power(1/2, sign(x(d)) * x(d));
%             p = p + 1; phis{p} = @(x) power(1, x(d));
            p = p + 1; phis{p} = @(x) power(2, x(d));
            p = p + 1; phis{p} = @(x) power(3, x(d));
%             p = p + 1; phis{p} = @(x) power(5, x(d));
        end
    end
end
        
function XPhi = mapPhis(phis, X)
    N = size(X, 1);
    M = numel(phis);

    XPhi = zeros(N, M);
    for n = 1:N
        for m = 1:M
            x = X(n, :);
            XPhi(n, m) = phis{m}(x);
        end
    end
end

