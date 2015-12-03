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

Tr = [];
Te = [];

% NOTE: you should do this randomly! and k-fold!
idx = randperm(size(train.X_hog,1));
mid = floor(length(idx)/2);
Tr.idxs = idx(1:mid);
%Tr.X = train.X_hog(Tr.idxs,:);
Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.idxs = idx(mid+1:end);
%Te.X = train.X_hog(Te.idxs,:);
Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);

clear idx mid;

%%
fprintf('Training simple neural network..\n');

addpath(genpath('toolboxs/DeepLearnToolbox-master/'));
addpath(genpath('our-code/'));


rng(8339);  % fix seed, this    NN may be very sensitive to initialization

% setup NN. The first layer needs to have number of features neurons,
%  and the last layer the number of classes (here four).
nn = nnsetup([size(Tr.X,2) 100 4]);
opts.numepochs =  20;  %  Number of full sweeps through data
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
LL = [1*(Tr.y == 1), ...
      1*(Tr.y == 2), ...
      1*(Tr.y == 3), ...
      1*(Tr.y == 4) ];  % first column, p(y=1)
                        % second column, p(y=2), etc

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

% get overall error [NOTE!! this is not the BER, you have to write the code
%                    to compute the BER!]
predErr = sum( classVote ~= Te.y ) / length(Te.y);
fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );

berErr = BER(Te.y, classVote);
fprintf('\nBER Testing error: %.2f%%\n\n', berErr * 100 );


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
