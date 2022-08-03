close all; clear; clc;
addpath(genpath('./HSCH/'));

param.db_name ='MIRFLICKR';
param.nbits = 4;

param.top_K = 2000;
param.pr_ind = [1:50:1000,1000];
param.pn_pos = [1:100:2000,2000];


fprintf('========%s start======== \n', 'HSCH');
[XTrain,YTrain,LTrain,XTest,YTest,LTest] = load_offline_data(param.db_name);
evaluate_HSCH(XTrain,YTrain,LTrain,XTest,YTest,LTest,param);
clearvars -except param


fprintf('\n ========%s start======== \n', 'HSCH_on');
[XChunk,YChunk,LChunk,XTest,YTest,LTest] = load_online_data(param.db_name);
evaluate_OHSCH(XChunk,YChunk,LChunk,XTest,YTest,LTest,param);
