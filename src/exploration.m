% Load data
load('HaNoi_classification.mat');
load('HaNoi_regression.mat');

% Display all plots
nbDim = size(X_train, 2);
plotDim = [4, 4];
plotPerFig = (plotDim(1)*plotDim(2));
nbFig = ceil(size(X_train, 2) / plotPerFig);

assert(nbDim < nbFig * plotPerFig);

for figNo = 0:(nbFig - 1)
    fig = figure;

    for subplotNo = 1:plotPerFig
        plotNo = figNo * plotPerFig + subplotNo;
        
        if plotNo <= nbDim
            subplot(plotDim(1), plotDim(2), subplotNo);
            hold on;

            plot(X_train(:, plotNo), y_train, '.');
            title(['dim ' num2str(plotNo)]);
        end
    end
    hold off;
end

% Display interesting plots
groupA = [25, 62]; % Features with big impacts on output
groupB = [9, 11, 15, 22, 27, 30, 38, 40, 44, 47, 56, 61]; % Discrete features
