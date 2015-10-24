function [cluster] = dummyEncodingTransformation(cluster)
%DUMMYENCODINGTRANSFORMATION Encode (dummily) categorical features
%
    cluster.train.X = dummyEncoding(cluster.train.X, 5);
    cluster.test.X = dummyEncoding(cluster.test.X, 5);

end

