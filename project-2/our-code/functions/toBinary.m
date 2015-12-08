function y = toBinary(y)
%
% Convert 4-class y {1, 2, 3, 4} to binary y {0, 1}
% Map 4 to 1 and {1, 2, 3} to -> 0
    otherIdx = (y == 4);
    y(otherIdx) = 0;
    y(~otherIdx) = 1;
end