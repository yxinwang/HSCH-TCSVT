function eva = evaluate_OHSCH(XChunk,YChunk,LChunk,XTest,YTest,LTest,param)

    % parameters
    param.etaX = 0.5; param.etaY = 0.5;
    param.iter = 5; param.sf = 0.05;
    param.omega = 10; param.beta = 0.01;
    param.theta = 1; param.lambda = 0.001;
    param.nchunks = length(XChunk);

    eva = cell(length(XChunk),1); Aux = {}; B = [];

    for chunki = 1:param.nchunks
        fprintf('.....chunk%3d..... \n', chunki);
        LTrain = cell2mat(LChunk(1:chunki,:));
        XTrain_new = XChunk{chunki,:};
        YTrain_new = YChunk{chunki,:};
        LTrain_new = LChunk{chunki,:};

        % normalize
        XTrain_new = NormalizeFea(XTrain_new,1);
        YTrain_new = NormalizeFea(YTrain_new,1);
        NLTrain_new = NormalizeFea(LTrain_new,1);

        % training
        [B_new,XW,YW,Aux] = train_OHSCH(XTrain_new,YTrain_new,LTrain_new,NLTrain_new,Aux,param);
        B = [B; B_new];
        
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
end