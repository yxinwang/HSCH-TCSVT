function [B,XW,YW] = train_HSCH(XTrain,YTrain,LTrain,NLTrain,param)
    
    % parameters
    etaX = param.etaX;
    etaY = param.etaY;
    max_iter = param.iter;
    kdim = param.nbits/param.sf;
    omega = param.omega;
    theta = param.theta;
    beta = param.beta;
    lambda = param.lambda;
    nbits = param.nbits;
    sel_num = 1000;
    if strcmp(param.db_name, 'NUS-WIDE')
        sel_num = 5000;
    end
    
    n = size(LTrain,1);
    dL = size(LTrain,2);
    dX = size(XTrain,2);
    dY = size(YTrain,2);
    
    a = (dL*(dL+2)+dL*sqrt(dL*(dL+2)))/4+eps;
    
    
    % hash code learning
    NXTrain = NormalizeFea(XTrain,1);
    NYTrain = NormalizeFea(YTrain,1);
    if kdim < n
        H = sqrt(n*nbits*nbits/kdim)*orth(rand(n,kdim));
        B = rsign(H,nbits);
        for i = 1:max_iter
            % update H
            Z = omega*B+nbits/(a+etaX+etaY)*(a*(NLTrain*(NLTrain'*B))+etaX*(NXTrain*(NXTrain'*B))...
                    +etaY*(NYTrain*(NYTrain'*B)));
            [~,Lmd,VV] = svd(Z'*Z);
            index = (diag(Lmd)>1e-6);
            V = VV(:,index); V_ = orth(VV(:,~index));
            U = Z *  (V / (sqrt(Lmd(index,index))));
            U_ = orth(randn(n,kdim-length(find(index==1))));
            H = sqrt(n*nbits*nbits/kdim)*[U U_]*[V V_]';

            % update B
            B = rsign(omega*H+nbits/(a+etaX+etaY)*(a*(NLTrain*(NLTrain'*H))+etaX*(NXTrain*(NXTrain'*H))...
                    +etaY*(NYTrain*(NYTrain'*H)))...
                    +theta*n*nbits/kdim*ones(n,kdim),nbits); %bit balance (extra)
        end
        
    else
        H = sqrt(n*nbits*nbits/kdim)*orth(rand(kdim,n))';
        B = rsign(H,nbits);
        for i = 1:max_iter
            % update H
            Z = omega*B+nbits/(a+etaX+etaY)*(a*(NLTrain*(NLTrain'*B))+etaX*(NXTrain*(NXTrain'*B))...
                    +etaY*(NYTrain*(NYTrain'*B)));
            [UU,Lmd,VV] = svd(Z,0);
            index = (diag(Lmd)>1e-6);
            V = VV(:,index); V_ = orth(VV(:,~index)); clear VV
            U = UU(:,index); U_ = orth(UU(:,~index)); clear UU
            H = sqrt(n*nbits*nbits/kdim)*[U U_]*[V V_]';
            
            % update B
            B = rsign(omage*H+nbits/(a+etaX+etaY)*(a*(NLTrain*(NLTrain'*H))+etaX*(NXTrain*(NXTrain'*H))...
                    +etaY*(NYTrain*(NYTrain'*H)))...
                    +theta*n*nbits/kdim*ones(n,kdim),nbits);
        end
    end
    clear Z Temp Lmd VV index U U_ V V_
    
    
    % hash function learning
    sel_idx = randperm(size(LTrain,1),sel_num);
    Bs = B(sel_idx,:);
    
    XW = (XTrain'*XTrain+lambda*eye(dX))\(XTrain'*B+((XTrain'*NLTrain)*NLTrain(sel_idx,:)'*a+...
        (XTrain'*NXTrain)*NXTrain(sel_idx,:)'*etaX+(XTrain'*NYTrain)*NYTrain(sel_idx,:)'*etaY)*Bs*beta*nbits/(a+etaX+etaY))...
        /(eye(kdim)+Bs'*Bs*beta);
    
    YW = (YTrain'*YTrain+lambda*eye(dY))\(YTrain'*B+((YTrain'*NLTrain)*NLTrain(sel_idx,:)'*a+...
        (YTrain'*NXTrain)*NXTrain(sel_idx,:)'*etaX+(YTrain'*NYTrain)*NYTrain(sel_idx,:)'*etaY)*Bs*beta*nbits/(a+etaX+etaY))...
        /(eye(kdim)+Bs'*Bs*beta);

end