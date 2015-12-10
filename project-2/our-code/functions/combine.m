function C = combine(a, b)
%
% Return a matrix C of all combinations of the elements of a and b
%
% See http://www.mathworks.com/matlabcentral/answers/98191-how-can-i-obtain-all-possible-combinations-of-given-vectors-in-matlab

    [A,B] = meshgrid(a, b);
    C     = cat(2, A', B');
    C     = reshape(C, [], 2);

end

