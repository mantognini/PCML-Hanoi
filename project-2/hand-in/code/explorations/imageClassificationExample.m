clearvars;

% -- GETTING STARTED WITH THE IMAGE CLASSIFICATION DATASET -- %

% Load features and labels of training data
addpath(genpath('data/train/'));
load 'data/train/train.mat';

%% --browse through the images and look at labels
addpath(genpath('toolboxs/piotr/'));
for i=1:10
    clf();
    
    % load img
    img = imread( sprintf('train/imgs/train%05d.jpg', i) );

    % show img
    subplot(131);
    imshow(img);
    title(sprintf('%d-th image; Label %d', i, train.y(i)));

    % show hog features analysis
    subplot(132);
    feature = hog( single(img)/255, 17, 8);
    im( hogDraw(feature) ); colormap gray;
    axis off; colorbar off;
    
    subplot(133);
    f = train.X_hog(i, :);
    f = reshape(f, 13, 13, 32); % Those dimensions correspond to hog( single(img)/255, 17, 8);
    im( hogDraw(f) ); colormap gray;
    axis off; colorbar off;

    pause;  % wait for key
end
clear i f img;

%% -- Example: split half and half into train/test, use HOG features
fprintf('Splitting into train/test..\n');

isBinary = 1;
isCNN = 1;

if isCNN
    X = sparse(double(train.X_cnn));
else
    X = train.X_hog;
end

Tr = [];
Te = [];

% NOTE: you should do this randomly! and k-fold!
idx = randperm(size(X,1));
mid = floor(length(idx)/2);
Tr.idxs = idx(1:mid);
Tr.X = X(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.idxs = idx(mid+1:end);
Te.X = X(Te.idxs,:);
Te.y = train.y(Te.idxs);


if isBinary
    Te.y = toBinary(Te.y);
    Tr.y = toBinary(Tr.y);
    fprintf('using binary prediction...\n');
end

clear idx mid;

%%
fprintf('Training simple neural network..\n');

addpath(genpath('toolboxs/DeepLearnToolbox-master/'));
addpath(genpath('our-code/'));


rng(8339);  % fix seed, this    NN may be very sensitive to initialization

% setup NN. The first layer needs to have number of features neurons,
%  and the last layer the number of classes (here four).
if isBinary
    nbOutput = 2;
else
    nbOutput = 4;
end

if isCNN
    nbHidden = 100;
else
    nbHidden = 10;
end

nn = nnsetup([size(Tr.X,2) nbHidden nbOutput]);
if isCNN
    opts.numepochs =  20;  %  Number of full sweeps through data
else
    opts.numepochs =  40;  %  Number of full sweeps through data
end
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

% WARNING: numepochs or batchsize too big seems to overfit!

% if == 1 => plots trainin error as the NN is trained
opts.plot               = 1;

nn.learningRate = 2;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
Tr.X = Tr.X(1:numSampToUse,:);
Tr.y = Tr.y(1:numSampToUse);

% normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std

% prepare labels for NN
if isBinary
    LL = [ 1*(Tr.y == 0), ... % either class 1, 2 or 3
           1*(Tr.y == 1) ];   % class 4 only
else
    LL = [ 1*(Tr.y == 1), ... % first column, p(y=1)
           1*(Tr.y == 2), ... % second column, p(y=2), etc
           1*(Tr.y == 3), ...
           1*(Tr.y == 4) ];
end

[nn, L] = nntrain(nn, Tr.normX, LL, opts);


Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

% to get the scores we need to do nnff (feed-forward)
%  see for example nnpredict().
% (This is a weird thing of this toolbox)
nn.testing = 1;
nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
nn.testing = 0;


% predict on the test set
nnPred = nn.a{end};

% get the most likely class
[~,classVote] = max(nnPred,[],2);
if isBinary
    classVote = classVote - 1; % map to {0, 1} and not {1, 2}
end

% get overall error [NOTE!! this is not the BER, you have to write the code
%                    to compute the BER!]
% predErr = sum( classVote ~= Te.y ) / length(Te.y);
% fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );

berErr = BER(Te.y, classVote);
if isBinary
    fprintf('\nbinary BER Testing error: %.2f%%\n\n', berErr * 100);
else
    fprintf('\nmulticlass BER Testing error: %.2f%%\n\n', berErr * 100);
end

if isCNN
    str = 'using CNN features';
else
    str = 'using HOG features';
end
figure('Name', str);
subplot(121);
imagesc(classVote); colorbar;
title('predictions');
subplot(122);
imagesc(Te.y); colorbar;
title('reality');

%% visualize samples and their predictions (test set)
figure;
for i=20:30  % just 10 of them, though there are thousands
    clf();

    img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
    imshow(img);


    % show if it is classified as pos or neg, and true label
    title(sprintf('Label: %d, Pred: %d', train.y(Te.idxs(i)), classVote(i)));

    pause;  % wait for key
end
