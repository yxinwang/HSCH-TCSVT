function [XChunk,YChunk,LChunk,XTest,YTest,LTest] = load_online_data(db_name)


load(['./datasets/',db_name,'.mat']);

if strcmp(db_name, 'IAPRTC-12')
    param.chunksize = 2000;
    clear V_tr V_te
    X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];%nonnegative features

elseif strcmp(db_name, 'MIRFLICKR')
    param.chunksize = 2000;
    X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];%nonnegative features

elseif strcmp(db_name, 'NUSWIDE10')
    param.chunksize = 10000;
    X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];%nonnegative features
    
elseif strcmp(db_name, 'MIRFLICKR_deep')
    param.chunksize = 2000;
    X = (X-min(min(X)))/(max(max(X))-min(min(X)));
end
clear I_tr I_te L_tr L_te T_tr T_te V_tr V_te XAll Y_pca

R = randperm(size(L,1));
queryInds = R(1:2000);
sampleInds = R(2001:end);
param.nchunks = floor(length(sampleInds)/param.chunksize);

XChunk = cell(param.nchunks,1);
YChunk = cell(param.nchunks,1);
LChunk = cell(param.nchunks,1);
for subi = 1:param.nchunks-1
    XChunk{subi,1} = X(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
    YChunk{subi,1} = Y(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
    LChunk{subi,1} = L(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
end
XChunk{param.nchunks,1} = X(sampleInds(param.chunksize*(param.nchunks-1)+1:end),:);
YChunk{param.nchunks,1} = Y(sampleInds(param.chunksize*(param.nchunks-1)+1:end),:);
LChunk{param.nchunks,1} = L(sampleInds(param.chunksize*(param.nchunks-1)+1:end),:);

XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);

end