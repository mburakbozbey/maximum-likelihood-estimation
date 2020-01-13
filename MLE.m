clear all;
clc;

dataset = readmatrix('path/to/dataset/');

% Variables and placeholders are defined.

pc = 0.75 ;     % percentage 75%
rng('default')  % For reproducibility
classNum = 8;

dimReduced = 20;
dim = dimReduced;

means = zeros(classNum, dim, 6);
C = zeros(dim, dim, classNum, 6);
priors = zeros(classNum, 6);

D = [];
T = [];

% Data preprocessing w.r.t the given dataset, please change these lines.
datasetClass = dataset(:,1);
datasetPatterns = dataset(:,2:end-1);

% Principal component analysis, applied separately on each feature type
dataPCA = zeros(length(datasetPatterns), dim);
for k=1:6
    [U, V] = pca(datasetPatterns(:,100*(k-1)+1:100*k));
    dataPCA(:,(k-1)*dimReduced+1:dimReduced*k) = datasetPatterns(:,100*(k-1)+1:100*k)*U(:,1:dimReduced);
end

dataset = zeros(length(datasetClass),6*dimReduced+1);
dataset(:,1) = datasetClass;
dataset(:,2:end) = dataPCA;

% Split each class, %75 to D, %25 to T
for k=1:8
    idx = dataset(:,1) == k;
    sampleC = dataset(idx,:);
    train = sampleC(1:round(pc*size(sampleC,1)),:);  % Split D set for kth class
    test = sampleC(round(pc*size(sampleC,1)):end,:); % Split T set for kth class
    D = [D;train];
    T = [T;test];
end

% Split patterns and classes of D & T set
dataClass = D(:,1);
dataPatterns = D(:,[2:end]);
dataRowCount = size(dataPatterns, 1);

% MLE
testClass = T(:,1);
testPatterns = T(:,[2:end]);
testRowCount = size(testPatterns, 1);

% Calculate prior probability, sample mean and covariance matrices for each
% class
for k=1:8
    for m=1:6
        numDataset = numel(D(:,[2:end]));
        idx = D(:,1) == k;
        sampleC = D(idx,1+(m-1)*dimReduced:m*dimReduced+1);
        samplePatterns = sampleC(:,[2:end]);
        priors(k, m) = length(samplePatterns)/length(D);                        % Priors
        means(k,:,m)= mean(samplePatterns);                                     % Sample mean
        C(:,:,k,m) = cov(samplePatterns) + 0.000001*eye(dimReduced,dimReduced); % Sample covariance
    end
end

predScore = [];
predClass = [];

% Apply simplified bayes rule to T set
for j=1:testRowCount
    P = zeros(8, 6);
    for m=1:6
        x = testPatterns(j,1+(m-1)*dimReduced:dimReduced*m);
        for c=1:8
            pdf = mvnpdf(x, means(c,:,m),C(:,:,c,m));
            P(c, m) =  priors(c,m)*log(pdf + 0.000001);
        end 
    end
    
    [score idx] = max(sum(P'));
    predScore = [predScore;score];
    predClass = [predClass;idx];
end

% Create confusion matrix and calculate precision & recall for T set

confM = confusionmat(testClass, predClass);
confM = confM';
confM = confM + eps; % Addition by epsilon due to division by zero error, etc.
testPrecision = diag(confM)./sum(confM,2);
testRecall = diag(confM)./sum(confM,1)';
accuracy = 1 - length(find(testClass~=predClass))/testRowCount; 
disp("Accuracy:");
disp(accuracy);
