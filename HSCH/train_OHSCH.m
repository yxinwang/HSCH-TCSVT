function [B_new,XW,YW,Aux] = train_OHSCH(XTrain_new,YTrain_new,LTrain_new,NLTrain_new,Aux,param)
    
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
    sel_num = floor(1000/param.nchunks);
    if strcmp(param.db_name, 'NUS-WIDE')
        sel_num = floor(5000/param.nchunks);
    end
    
    n = size(LTrain_new,1);
    dL = size(LTrain_new,2);
    dX = size(XTrain_new,2);
    dY = size(YTrain_new,2);
    
    a = (dL*(dL+2)+dL*sqrt(dL*(dL+2)))/4+eps;
    
    
    % hash code learning
    NXTrain_new = NormalizeFea(XTrain_new,1);
    NYTrain_new = NormalizeFea(YTrain_new,1);
    if kdim < n
        H_new = sqrt(n*nbits*nbits/kdim)*orth(rand(n,kdim));
        B_new = rsign(H_new,nbits);
        for i = 1:max_iter
            if isempty(Aux)
                CC{1,1} = NLTrain_new'*B_new;
                JJ{1,1} = NXTrain_new'*B_new;
                JJ{1,2} = NYTrain_new'*B_new;
            else
                CC{1,1} = Aux.CC{1,1} + NLTrain_new'*B_new;
                JJ{1,1} = Aux.JJ{1,1} + NXTrain_new'*B_new;
                JJ{1,2} = Aux.JJ{1,2} + NYTrain_new'*B_new;
            end

            % update H
            Z = omega*B_new+nbits/(a+etaX+etaY)*(a*(NLTrain_new*CC{1,1})+etaX*(NXTrain_new*JJ{1,1})...
                +etaY*(NYTrain_new*JJ{1,2}));
            [~,Lmd,VV] = svd(Z'*Z);
            index = (diag(Lmd)>1e-6);
            V = VV(:,index); V_ = orth(VV(:,~index));
            U = Z *  (V / (sqrt(Lmd(index,index))));
            U_ = orth(randn(n,kdim-length(find(index==1))));
            H_new = sqrt(n*nbits*nbits/kdim)*[U U_]*[V V_]';
            
            if isempty(Aux)
                CC{1,2} = NLTrain_new'*H_new;
                KK{1,1} = NXTrain_new'*H_new;
                KK{1,2} = NYTrain_new'*H_new;
            else
                CC{1,2} = Aux.CC{1,2} + NLTrain_new'*H_new;
                KK{1,1} = Aux.KK{1,1} + NXTrain_new'*H_new;
                KK{1,2} = Aux.KK{1,2} + NYTrain_new'*H_new;
            end
            
            % update B
            B_new = rsign(omega*H_new+nbits/(a+etaX+etaY)*(a*(NLTrain_new*CC{1,2})+etaX*(NXTrain_new*KK{1,1})...
                +etaY*(NYTrain_new*KK{1,2}))...
                +theta*n*nbits/kdim*ones(n,kdim),nbits);
        end
        
        
    else
        H_new = sqrt(n*nbits*nbits/kdim)*orth(rand(kdim,n))';
        B_new = rsign(H_new,nbits);
        for i = 1:max_iter            
            if isempty(Aux)
                CC{1,1} = NLTrain_new'*B_new;
                JJ{1,1} = NXTrain_new'*B_new;
                JJ{1,2} = NYTrain_new'*B_new;
            else
                CC{1,1} = Aux.CC{1,1} + NLTrain_new'*B_new;
                JJ{1,1} = Aux.JJ{1,1} + NXTrain_new'*B_new;
                JJ{1,2} = Aux.JJ{1,2} + NYTrain_new'*B_new;
            end

            % update H
            Z = omega*B_new+nbits/(a+etaX+etaY)*(a*(NLTrain_new*CC{1,1})+etaX*(NXTrain_new*JJ{1,1})...
                +etaY*(NYTrain_new*JJ{1,2}));
            [UU,Lmd,VV] = svd(Z,0);
            index = (diag(Lmd)>1e-6);
            V = VV(:,index); V_ = orth(VV(:,~index)); clear VV
            U = UU(:,index); U_ = orth(UU(:,~index)); clear UU
            H_new = sqrt(n*nbits*nbits/kdim)*[U U_]*[V V_]';
            
            if isempty(Aux)
                CC{1,2} = NLTrain_new'*H_new;
                KK{1,1} = NXTrain_new'*H_new;
                KK{1,2} = NYTrain_new'*H_new;
            else
                CC{1,2} = Aux.CC{1,2} + NLTrain_new'*H_new;
                KK{1,1} = Aux.KK{1,1} + NXTrain_new'*H_new;
                KK{1,2} = Aux.KK{1,2} + NYTrain_new'*H_new;
            end
            
            % update B
            B_new = rsign(omega*H_new+nbits/(a+etaX+etaY)*(a*(NLTrain_new*CC{1,2})+etaX*(NXTrain_new*KK{1,1})...
                +etaY*(NYTrain_new*KK{1,2}))...
                +theta*n*nbits/kdim*ones(n,kdim),nbits);
        end
    end
    clear Z Temp Lmd VV index U U_ V V_
    
    
    % hash function learning
    sel_idx = randperm(size(LTrain_new,1),sel_num);
    Bs_new = B_new(sel_idx,:);
    
    Ss_new = (NLTrain_new*NLTrain_new(sel_idx,:)'*a+...
        NXTrain_new*NXTrain_new(sel_idx,:)'*etaX+NYTrain_new*NYTrain_new(sel_idx,:)'*etaY)/(a+etaX+etaY);

    if isempty(Aux)
        Aux.CC{1,1} = NLTrain_new'*B_new;
        Aux.CC{1,2} = NLTrain_new'*H_new;
        Aux.JJ{1,1} = NXTrain_new'*B_new;
        Aux.JJ{1,2} = NYTrain_new'*B_new;
        Aux.KK{1,1} = NXTrain_new'*H_new;
        Aux.KK{1,2} = NYTrain_new'*H_new;
        
        Aux.CC{1,3} = Bs_new'*Bs_new;
        Aux.EE{1,1} = XTrain_new'*Ss_new*Bs_new;
        Aux.EE{1,2} = YTrain_new'*Ss_new*Bs_new;
        Aux.FF{1,1} = XTrain_new'*B_new;
        Aux.FF{1,2} = YTrain_new'*B_new;
        Aux.GG{1,1} = XTrain_new'*XTrain_new;
        Aux.GG{1,2} = YTrain_new'*YTrain_new;
        
    else
        Aux.CC{1,1} = Aux.CC{1,1} + NLTrain_new'*B_new;
        Aux.CC{1,2} = Aux.CC{1,2} + NLTrain_new'*H_new;
        Aux.JJ{1,1} = Aux.JJ{1,1} + NXTrain_new'*B_new;
        Aux.JJ{1,2} = Aux.JJ{1,2} + NYTrain_new'*B_new;
        Aux.KK{1,1} = Aux.KK{1,1} + NXTrain_new'*H_new;
        Aux.KK{1,2} = Aux.KK{1,2} + NYTrain_new'*H_new;
        
        Aux.CC{1,3} = Aux.CC{1,3} + Bs_new'*Bs_new;
        Aux.EE{1,1} = Aux.EE{1,1} + XTrain_new'*Ss_new*Bs_new;
        Aux.EE{1,2} = Aux.EE{1,2} + YTrain_new'*Ss_new*Bs_new;
        Aux.FF{1,1} = Aux.FF{1,1} + XTrain_new'*B_new;
        Aux.FF{1,2} = Aux.FF{1,2} + YTrain_new'*B_new;
        Aux.GG{1,1} = Aux.GG{1,1} + XTrain_new'*XTrain_new;
        Aux.GG{1,2} = Aux.GG{1,2} + YTrain_new'*YTrain_new;
    end
    
    XW = (Aux.GG{1,1}+lambda*eye(dX))\(Aux.FF{1,1}+Aux.EE{1,1}*beta*nbits)...
        /(eye(kdim)+Aux.CC{1,3}*beta);
    YW = (Aux.GG{1,2}+lambda*eye(dY))\(Aux.FF{1,2}+Aux.EE{1,2}*beta*nbits)...
        /(eye(kdim)+Aux.CC{1,3}*beta);
end