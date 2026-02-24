function [Mean] = tenFoldResult(features,labels,idx,seed)

[n,d]=size(features);
c = size(labels,2);

if d<=100
    s = d*0.4;
elseif d<=500
    s = d*0.3;
elseif d<=1000
    s = d*0.2;
else
    s = d*0.1;
end
s = round(s);
idx = idx(1:s);

rand('seed',seed);
fold = buffer(randperm(n),10);
fold_ = cell(10,1);
for i=1:10
    foldi = fold(i,:);
    foldi = foldi(foldi ~= 0);
    fold_{i} = foldi;
end
result = zeros(10,6);
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','MaxIterations',500000,...
                        'FiniteDifferenceStepSize',1e-20,'SpecifyObjectiveGradient',true);
features = features(:,idx);     
item=zeros(s,c);
for i=1:10
    testIndex = fold_{i};
    trainingIndex = setdiff((1:n),testIndex);
    global features_i labels_i
    features_i = features(trainingIndex,:);
    labels_i = labels(trainingIndex,:);
    weights_i = fminunc(@bfgsProcess,item,options);
    preDistribution_i = lldPredict(weights_i,features(testIndex,:));
    [~,measures_i] = computeMeasures(labels(testIndex,:),preDistribution_i);
    result(i,:) = measures_i;
end
Mean = sum(result)/10;
end

