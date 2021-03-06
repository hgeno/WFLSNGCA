function [g,theta,hparams,MedDim,C,cind]=LSLDG(X,op,C,MedDim)
%
% Estimating Log-Density Gradients(multi-band)
% 
% X: (sample) by (dim) matrix
% op: options
% C: center points of kernels
%

narginchk(2,4);

[op.dim,op.samples]=size(X);

% if isfield(op,'rand_id')
%     s = RandStream('mt19937ar','Seed',op.rand_id);
%     RandStream.setGlobalStream(s);
% end

if ~isfield(op,'dim') || ~isfield(op,'samples')
    [op.dim,op.samples]=size(X);
end

if ~isfield(op,'bnum')
    op.bnum=min(op.samples,100);
end

if ~isfield(op,'sigma_list')
    op.sigma_list=logspace(-1,1,10);
end

if ~isfield(op,'lambda_list')
    op.lambda_list=logspace(-5,1,10);
end

if ~isfield(op,'cvfold')
    op.cvfold=5;
end

% if ~isfield(op,'bfunc')
%     op.bfunc = 1;
% %     op.bfunc = 0; %GaussKernel
% end

if ~exist('C','var') || isempty(C)
    cind=randperm(op.samples,op.bnum);
    C=X(:,cind);
else
    cind=[];
end

%{
if ~exist('Nsamples_for_Med','var') || isempty(Nsamples_for_Med)
    Nsamples_for_Med = 1000;
end
%}
if ~exist('MedDim','var') || isempty(MedDim)
    MedDim=ones(op.dim,1);
%     if op.samples > Nsamples_for_Med
%         X_for_Med = X(:,randperm(Nsamples_for_Med)); 
%     else
%         X_for_Med = X;
%     end
% 
%     MedDim=MedianDiffDim(X_for_Med);
%     if ~isempty(find(MedDim<eps));
%         disp('median<eps');
%         MedDim(:)=MedianDiff(X_for_Med);
%     end
end

% MedDim=repmat(MedianDiff(X),[op.dim,1]);


% Difference to centers (size: bnum by samles by dim)
XC_diff=repmat(permute(C,[2,3,1]),[1,op.samples,1])...
    -repmat(permute(X,[3,2,1]),[op.bnum,1,1]);

% Distance from centers (size: bnum by samples)
XC_dist=sum(XC_diff.^2,3);

% cross validation
cv_fold=(1:op.cvfold);
cv_split=floor((0:op.samples-1)*op.cvfold./op.samples)+1;
cv_index=cv_split(randperm(op.samples));




%% cross validation
hparams.sigma=zeros(op.dim,1); hparams.lambda=zeros(op.dim,1);
hparams.sigma_id=zeros(op.dim,1); hparams.lambda_id=zeros(op.dim,1);

for dd=1:op.dim                
    score_cv=zeros(length(op.sigma_list),length(op.lambda_list),length(cv_fold));

    for sigma_index=1:length(op.sigma_list)
        sigma_tmp=MedDim(dd)*op.sigma_list(sigma_index);
%         sigma_tmp=op.sigma_list(sigma_index);

        GauKer=exp(-XC_dist/(2*sigma_tmp^2));        
        for kk=cv_fold  
            if op.bfunc == 1
                psi_train=XC_diff(:,cv_index~=kk,dd).*GauKer(:,cv_index~=kk);  
                psi_test=XC_diff(:,cv_index==kk,dd).*GauKer(:,cv_index==kk); 
                phi_train=(XC_diff(:,cv_index~=kk,dd).^2/sigma_tmp^2-1).*GauKer(:,cv_index~=kk);
                phi_test=(XC_diff(:,cv_index==kk,dd).^2/sigma_tmp^2-1).*GauKer(:,cv_index==kk);
            else
                psi_train = GauKer(:,cv_index~=kk); 
                phi_train=(XC_diff(:,cv_index~=kk,dd)/sigma_tmp^2).*GauKer(:,cv_index~=kk);
                psi_test=GauKer(:,cv_index==kk);      
                phi_test=(XC_diff(:,cv_index==kk,dd)/sigma_tmp^2).*GauKer(:,cv_index==kk);
            end
      
            K_train=psi_train*psi_train'/size(psi_train,2);
            K_test=psi_test*psi_test'/size(psi_test,2);
            h_train=mean(phi_train,2);
            h_test=mean(phi_test,2);

            for lambda_index=1:length(op.lambda_list)
                lambda_tmp=op.lambda_list(lambda_index);
                K_train_lam = K_train+lambda_tmp*eye(size(K_train));
                if rank(K_train_lam) == op.bnum
                    thetah=linsolve(K_train_lam,-h_train);

                    term1=thetah'*K_test*thetah;
                    term2=thetah'*h_test;                    

                    score_cv(sigma_index,lambda_index,kk)=term1+2*term2;
                else
                    fprintf('Inf:sigma=%g(id:%g), lambda=%g(id:%g)',sigma_tmp,sigma_index,lambda_tmp,lambda_index)
                    score_cv(sigma_index,lambda_index,kk)=Inf;
                end
            end %lambda                            
        end % kk
    end % sigma
        
    [score_cv_tmp,lambda_index]=min(mean(score_cv,3),[],2);
    if sum(score_cv_tmp(score_cv_tmp~=Inf))==0
        disp('g:Please change hyper-parameter list');
    end
    [~,hparams.sigma_id(dd)]=min(score_cv_tmp);
    hparams.sigma(dd)=MedDim(dd)*op.sigma_list(hparams.sigma_id(dd));
    hparams.lambda_id(dd) = lambda_index(hparams.sigma_id(dd));
    hparams.lambda(dd)=op.lambda_list(hparams.lambda_id(dd));   
%     sigma(dd)=op.sigma_list(sigma_index);
%     fprintf('dd=%g, sigma_id=%g, lambda_id=%g\n',dd,hparams.sigma_id(dd),hparams.lambda_id(dd));
    fprintf('dd=%g, sigma=%g(id:%g), lambda=%g(id:%g)\n',dd,hparams.sigma(dd),hparams.sigma_id(dd),hparams.lambda(dd),hparams.lambda_id(dd));
end % dd

%% compute theta
theta=zeros(op.bnum,op.dim);
for dd=1:op.dim
    GauKer=exp(-XC_dist/(2*hparams.sigma(dd)^2));

    if op.bfunc == 1
        psi=XC_diff(:,:,dd).*GauKer;
        phi=(XC_diff(:,:,dd).^2/hparams.sigma(dd)^2-1).*GauKer;
    else
        psi=GauKer;
        phi=(XC_diff(:,:,dd)/hparams.sigma(dd)^2).*GauKer;
    end
    K=psi*psi'/size(psi,2);
    h=mean(phi,2);
        
    theta(:,dd)=linsolve(K+hparams.lambda(dd)*eye(size(K)),-h);
end

%% compute grad 
GauKer3D=exp(-bsxfun(@times,XC_dist,1./(2*permute(hparams.sigma.^2,[2,3,1]))));

if op.bfunc == 1
    psi=XC_diff.*GauKer3D;
else
    psi=GauKer3D; %b*n*d
end
 
g=permute(sum(bsxfun(@times,theta,permute(psi,[1,3,2])),1),[2,3,1]);
