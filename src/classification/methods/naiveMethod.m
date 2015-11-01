function yValidPred = naiveMethod(~, ~, XValid)
% if 11th feature is <= 10 then classify as 1, else
% 50-50% chance.
%

    N = size(XValid, 1);
    yValidPred = (XValid(:, 11) > -10) + (XValid(:, 11) <= -10) .* binornd(1, 0.5, N, 1);

end

