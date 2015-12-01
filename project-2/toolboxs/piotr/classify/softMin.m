function M = softMin( D, sigma )
% Calculates the softMin of a vector.
%
% Let D be a vector.  Then the softMin of D is defined as:
%   s = exp(-D/sigma^2) / sum( exp(-D/sigma^2) )
% The softMin is a way of taking a dissimilarity (distance) vector D and
% converting it to a similarity vector s, such that sum(s)==1. If D is an
% NxK array, is is treated as N K-dimensional vectors, and the return is
% likewise an NxK array.  This is useful if D is a distance matrix,
% generated by the likes of pdist2.
%
% Note that as sigma->0, softMin's behavior tends toward that of the
% standard min function.  That is the softMin of a vector D has all zeros
% with a single 1 in the location of the smallest value of D. For example,
% "softMin([.2 .4 .1 .3],eps)" returns "[0 0 1 0]".  As sigma->inf, then
% softMin(D,sigma) tends toward "ones(1,n)/n", where n==length(D).
%
% If D contains the squared euclidean distance between a point y and k
% points xi, then there is a probabilistic interpretation for softMin.  If
% we think of the k points representing equal variant gaussians each with
% mean xi and std sigma, then the softMin returns the relative probability
% of y being generated by each gaussian.
%
% USAGE
%  M = softMin( D, sigma )
%
% INPUTS
%  D       - NxK dissimilarity matrix
%  sigma   - controls 'softness' of softMin
%
% OUTPUTS
%  M       - the softMin (indexes into D)
%
% EXAMPLE - 1
%  C = [0 0; 1 0; 0 1; 1 1]; x=[.7,.3; .1 .2];
%  D = pdist2( x, C ), M = softMin( D, .25 )
%
% EXAMPLE - 2
%  fplot( 'softMin( [0.5 0.2 .4], x )', [0 5] );
%  xlabel('sigma'); ylabel('assignments')
%
% See also PDIST2, SOFTMAX
%
% Piotr's Computer Vision Matlab Toolbox      Version 2.0
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

if( sigma==0 ) % special case, make fast
  [~, inds] = min(D,[],2); [n, k] = size(D);
  M = subsToArray( [(1:n)' inds], ones(n,1), [n k] );

else % general case
  M = exp( -D / sigma^2 );
  M(isinf(M))=1e50;
  sumM = sum( M, 2 );
  sumMzero = (sumM==0);
  if( any(sumMzero) )
    [~, inds] = min(D,[],2); [n, k] = size(D);
    Mhard = subsToArray( [(1:n)' inds], ones(n,1), [n k] );
    M( sumMzero, : ) = Mhard( sumMzero, : );
    sumM = sum( M, 2 );
  end
  M = M ./ sumM( :, ones(1,size(M,2)) );
end
