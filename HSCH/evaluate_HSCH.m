function evaluation_info=evaluate_HSCH(XTrain,YTrain,LTrain,XTest,YTest,LTest,param)

    % parameters
    param.etaX = 0.5; param.etaY = 0.5;
    param.iter = 5; param.sf = 0.05;
    param.omega = 10; param.beta = 0.01;
    param.theta = 10; param.lambda = 0.001;
    
    % normalize
    XTrain = NormalizeFea(XTrain,1);
    YTrain = NormalizeFea(YTrain,1);
    NLTrain = NormalizeFea(LTrain,1);
    XTest = NormalizeFea(XTest,1);
    YTest = NormalizeFea(YTest,1);
    
    % training
    [B,XW,YW] = train_HSCH(XTrain,YTrain,LTrain,NLTrain,param);
    
    % evaluate
    BxTest = rsign(XTest*XW,param.nbits);
    ByTest = rsign(YTest*YW,param.nbits);
    DHamm = BxTest*B';
    [~, orderH] = sort(DHamm,2,'descend');
    evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
    fprintf('Image_VS_Text_MAP: %f.\n', evaluation_info.Image_VS_Text_MAP);
    DHamm = ByTest*B';
    [~, orderH] = sort(DHamm,2,'descend');
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    fprintf('Text_VS_Image_MAP: %f.\n', evaluation_info.Text_VS_Image_MAP);
end