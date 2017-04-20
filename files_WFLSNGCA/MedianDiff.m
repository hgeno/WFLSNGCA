function s=MedianDiff(X)

n=size(X,2);

XX_diff=repmat(permute(X,[2,3,1]),[1,n,1])...
    -repmat(permute(X,[3,2,1]),[n,1,1]);

s=median(abs(XX_diff(:)));

