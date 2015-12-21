function y = toBinary(y, optClass)
%
% Convert 4-class y {1, 2, 3, 4} to binary y {0, 1}

    if (nargin == 1)
        % By default map 4 to 1 and {1, 2, 3} to -> 0
        class = 4;
    else
        class = optClass;
    end

    otherIdx = (y == class);
    y( otherIdx) = 0;
    y(~otherIdx) = 1;

end
