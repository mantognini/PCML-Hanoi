%%
clear all;

%% Concrete case from 
% http://icapeople.epfl.ch/mekhan/pcml15/project-2/objectDetection.html

negSamples = ones(90, 1) * (-1); % 90% negative samples
posSamples = ones(10, 1); % 10% positive samples

naivePredictions = ones(100, 1) * (-1);
cunningPredictions = ones(100, 1);


assert(BER([negSamples; posSamples], naivePredictions) == 0.5);
assert(BER([negSamples; posSamples], cunningPredictions) == 0.5);