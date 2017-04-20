function [ngmatrix,projdata,projmat,hparams,Ncond,y,g,ctime_g]=LSNGCA(X,dz,op)

disp('LSNGCA')

[op.dim,op.samples]=size(X);
X=bsxfun(@minus,X,mean(X,2));
[X,wMat,Ncond] = whitening(X);
% Err_whitening = norm(wMat^2*covMatrix-eye(op.dx),'fro');

ticID = tic();
[g,~,hparams]=LSLDG(X,op);
ctime_g = toc(ticID);
y=g+X;

[E,D]=eig(y*y'/op.samples); 

dd=diag(D);
[~,ind]=sort(dd,'descend');
B=E(:,ind(1:dz));

ev = (wMat')*B;
% ngmatrix=qr(ev,0);
ngmatrix=ev*inv(ev'*ev)^0.5;

projdata=B'*X;
projmat=B'*wMat; % whitening & projecting matrix 
end
