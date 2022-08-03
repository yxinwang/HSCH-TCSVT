function A = rsign(X,r)
    [n,d] = size(X); 
    if r == d
        A = sign(X);
    else
        [~,idx] = sort(X,2,'descend');
        tt = idx(:,1:r);
        A = sparse(repmat((1:n)',[r 1]),tt(:),1); A = full(A);
    end
end