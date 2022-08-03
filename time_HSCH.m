close all; clear; clc;
addpath(genpath('./HSCH/'));

param.db_name ='MIRFLICKR';
param.nbits = 4;
param.top_K = 2000;
param.pr_ind = [1:50:1000,1000];
param.pn_pos = [1:100:2000,2000];


fprintf('========%s start======== \n', 'HSCH');
[XTrain,YTrain,LTrain,XTest,YTest,LTest] = load_offline_data(param.db_name);

% normalize
XTrain = NormalizeFea(XTrain,1);
YTrain = NormalizeFea(YTrain,1);
NLTrain = NormalizeFea(LTrain,1);
XTest = NormalizeFea(XTest,1);
YTest = NormalizeFea(YTest,1);
[n,dX] = size(XTrain); dY = size(YTrain,2);
    
    
% training
tic;
HSCHparam = param;
HSCHparam.etaX = 0.5; HSCHparam.etaY = 0.5;
HSCHparam.iter = 5; HSCHparam.sf = 0.05;
HSCHparam.omega = 10; HSCHparam.beta = 0.01;
HSCHparam.theta = 10; HSCHparam.lambda = 0.001;
[B,XW,YW] = train_HSCH(XTrain,YTrain,LTrain,NLTrain,HSCHparam);
trainT = toc;
fprintf('trainT: %f.\n', trainT);


% query
tic;
[~,idx] = sort(XTest(1,:)*XW,2,'descend');
BxTest = idx(1:param.nbits);
[~,idx] = sort(YTest(1,:)*YW,2,'descend');
ByTest = idx(1:param.nbits);
DHamm1 = zeros(1,n); DHamm2 = zeros(1,n);
for iii = 1:n
    DHamm1(1,iii) = sum(B(iii,BxTest));
    DHamm2(1,iii) = sum(B(iii,ByTest));
end
[~, orderH1] = sort(DHamm1, 2);
[~, orderH2] = sort(DHamm2, 2);
queryT = toc;
fprintf('queryT: %f.\n', queryT);


fprintf('========%s start======== \n', 'Dense methods');

% training
XW = randn(dX,param.nbits);
YW = randn(dY,param.nbits);
B = sign(randn(n,param.nbits));


% test dense codes
tic;
BxTest = sign(XTest(1,:)*XW);
ByTest = sign(YTest(1,:)*YW);
DHamm1 = zeros(1,n); DHamm2 = zeros(1,n);
for iii = 1:n
    DHamm1(1,iii) = nnz(BxTest-B(iii,:));
    DHamm2(1,iii) = nnz(ByTest-B(iii,:));
end
[~, orderH1] = sort(DHamm1, 2);
[~, orderH2] = sort(DHamm2, 2);
queryT_dense =toc;
fprintf('queryT_dense: %f.\n', queryT_dense);
