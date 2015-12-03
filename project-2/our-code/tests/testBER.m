%%
clear all;

%% Basic cases
p1 = ones(50, 1);
p2 = zeros(50, 1);

eps = 0.0001;
assert(abs(BER([p1; p2], [p2; p1]) - 1) < eps); % all wrong
assert(abs(BER([p1; p2], [p1; p2]) - 0) < eps); % all correct

%% Concrete case from 
% http://icapeople.epfl.ch/mekhan/pcml15/project-2/objectDetection.html

negSamples = ones(90, 1) * (-1); % 90% negative samples
posSamples = ones(10, 1); % 10% positive samples

naivePredictions = ones(100, 1) * (-1);
cunningPredictions = ones(100, 1);

eps = 0.0001;
assert((BER([negSamples; posSamples], naivePredictions) - 0.5) < eps);
assert((BER([negSamples; posSamples], cunningPredictions) - 0.5) < eps);

%% Advanced cases

posSamples = ones(10, 1); % 10% positive samples
negSamples = zeros(90, 1); % 90% negative samples

n1 = 54; % 0.6 * 90
predNeg1 = zeros(n1, 1); % 60% of negative samples are correctly guessed
predNeg2 = ones(90 - 54, 1); % 40% are wrongly guessed

n2 = 6; % 0.6 * 10
predPos1 = ones(n2, 1); % 60% of positive samples are correctly guessed
predPos2 = zeros(4, 1); % 40% are wrongly guessed

eps = 0.0001;
ber = BER([negSamples; posSamples], [predNeg1; predNeg2; predPos1; predPos2]);
assert((ber - 0.4) < eps); % ber is average error percentage on each class

%% Multiclass case

p1 = ones(20, 1) * 1;
p2 = ones(30, 1) * 2;
p3 = ones(50, 1) * 3;

guess1Pos = ones(4, 1) * 1; % 20% are correct guess
guess1Neg = zeros(16, 1);

guess2Pos = ones(18, 1) * 2; % 60% are correct guess
guess2Neg = zeros(12, 1);

guess3Pos = ones(20, 1) * 3; % 40% are correct guess
guess3Neg = zeros(30, 1);

% Then (80+40+60)/3 = 180/3 = 60 % are wrong guess overall
eps = 0.0001;
ber = BER([p1; p2; p3], [guess1Pos; guess1Neg; guess2Pos; guess2Neg; guess3Pos; guess3Neg]);
assert((ber - 0.6) < eps); % ber is average error percentage on each class

