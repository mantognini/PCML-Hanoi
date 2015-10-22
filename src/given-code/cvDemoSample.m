function cvDemo()
% A demonstration of cross-validation
% The text written at top displayes, when you write  'help cvDemo'
% in Matlab command window. Try it out.
% Written by Emtiyaz, EPFL

% load data
clear all;
load('dataEx3.mat');

% choose degree
degree = 7;

% split data in K fold
setSeed(1);
K = 4;
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% For all lambda values
lambda = logspace(-4,2,30);
for i = 1:length(lambda)
    for k = 1:K
        % get k'th subgroup in test, others in train
        idxTe = idxCV(k,:);
        idxTr = idxCV([1:k-1 k+1:end],:);
        idxTr = idxTr(:);
        yTe = y(idxTe);
        XTe = X(idxTe,:);
        yTr = y(idxTr);
        XTr = X(idxTr,:);

        % form tX
        tXTr = [ones(length(yTr), 1) myPoly(XTr, degree)];
        tXTe = [ones(length(yTe), 1) myPoly(XTe, degree)];

        % least squares
        beta = ridgeRegression(yTr,tXTr, lambda(i));

        % train and test MSE
        mseTrSub(k) = sqrt(2*computeCost(yTr,tXTr,beta));
        mseTeSub(k) = sqrt(2*computeCost(yTe,tXTe,beta));

    end
    % compute mean
    mseTr(i) = mean(mseTrSub);
    mseTe(i) = mean(mseTeSub);
end

% plot
semilogx(lambda, mseTr,'b-o','linewidth', 2,'markerfacecolor', [1 1 1]);
hold on
semilogx(lambda, mseTe,'r-o','linewidth', 2,'markerfacecolor', [1 1 1]);
xlabel('lambda');
legend('Train error', 'test error', 'location', 'southeast');
grid on;
set(gca,'xticklabel',[],'yticklabel',[]);
