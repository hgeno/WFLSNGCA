function s=MedianDiffDim(X)

[dim,n]=size(X);

s=zeros(dim,1);
for dd=1:dim
    tmp=X(dd,:);
    
    s(dd)=median(reshape(abs(bsxfun(@minus,tmp',tmp)),[n^2,1]));    
end


% XX_diff=repmat(permute(X,[2,3,1]),[1,n,1])...
%     -repmat(permute(X,[3,2,1]),[n,1,1]);
% 
% s=median(reshape(abs(XX_diff(:)),[n^2,dim]),1);