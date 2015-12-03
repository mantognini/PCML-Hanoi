%%
clear all;

%% Basic cases
p1 = ones(50, 1);
p2 = zeros(50, 1);

assert(BER([p1; p2], [p2; p1]) == 1); % all wrong
assert(BER([p1; p2], [p1; p2]) == 0); % all correct

%% Concrete case from 
% http://icapeople.epfl.ch/mekhan/pcml15/project-2/objectDetection.html

negSamples = ones(90, 1) * (-1); % 90% negative samples
posSamples = ones(10, 1); % 10% positive samples

naivePredictions = ones(100, 1) * (-1);
cunningPredictions = ones(100, 1);


assert(BER([negSamples; posSamples], naivePredictions) == 0.5);
assert(BER([negSamples; posSamples], cunningPredictions) == 0.5);

%% Advanced cases

posSamples = ones(10, 1); % 10% positive samples
negSamples = zeros(90, 1); % 90% negative samples

n1 = 54; % 0.6 * 90
predNeg1 = zeros(n1, 1); % 60% of negative samples are correctly guessed
predNeg2 = ones(90 - 54, 1); % 40% are wrongly guessed

n2 = 6; % 0.6 * 10
predPos1 = ones(n2, 1); % 60% of positive samples are correctly guessed
predPos2 = zeros(4, 1); % 40% are wrongly guessed

ber = BER([negSamples; posSamples], [predNeg1; predNeg2; predPos1; predPos2]);
assert(ber == 0.4); % ber is average error percentage on each class

