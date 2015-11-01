function y = sigmToZeroOne(yProbs)
%
    y = (yProbs > (1 - yProbs)) + 0;
end