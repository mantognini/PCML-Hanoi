clear all;

M = 10000; % nb of observations
N = 3; % nb of features
A = zeros(M, N);
eps = 0.1;

for n = 1:N
    mu = floor(rand() * 20) - 10;
    sigma = rand() * 2.5 + 0.3;
    
    cMu = mu + 2 * eps;
    cSigma = sigma + 2 * eps;
    
    % generate the data for this feature
    % .. until it is approximatively correct
    while abs(mu - cMu) > eps || abs(sigma - cSigma) > eps
        A(:, n) = normrnd(mu, sigma, M, 1);
        cMu = mean(A(:, n));
        cSigma = std(A(:, n));
    end
end

% normalize the data
B = normalize(A);

rudeEps = 10^(-12);
for n = 1:N
    assert(abs(mean(B(:, n)) - 0) < rudeEps); % mean should now be 0
    assert(abs(std(B(:, n)) - 1) < rudeEps); % sigma shoulw now be 1
end

% Optionally, you may want to visualize
figure('Name', 'Before and after normalization');
for i = 1:2
    if i == 1
        C = A;
        titleS = 'before';
    else
        C = B;
        titleS = 'after';
    end
    
    % plot features
    subplot(2, 1, i);
    for n = 1:N
        histogram(C(:, n), 30);
        hold on;
    end
    xlabel(titleS);
    hold off;
end
