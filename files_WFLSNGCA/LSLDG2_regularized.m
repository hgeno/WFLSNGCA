function [g,nabla_g,theta,hparams,MedDim,C,cind]=LSLDG2_regularized(X,op,C,MedDim)
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

if ~isfield(op,'bfunc')
    op.bfunc = 1;
%     op.bfunc = 0; %GaussKernel
end

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
%     MedDim=MedianDiffDim(min(1000,op.samples));
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
% op.bfunc =2;

if op.bfunc ~= 1
    XC_dist=sum(XC_diff.^2,3);
end

X_XC_diff = sum(bsxfun(@times,permute(X,[3,2,1]),XC_diff),3);
X_repmat = repmat(permute(X,[3,2,1]),[op.bnum,1,1]);%b*n*d

CovX = cov(X');
CovX_inv = inv(CovX);

if op.bfunc == 1
    XC_dist=sum(sum(bsxfun(@times,XC_diff,permute(CovX_inv,[3,4,1,2])),4).*XC_diff,3);
%     MedDim = CovX_inv*MedDim;
end
CovX_XC_diff = sum(bsxfun(@times,permute(CovX_inv,[3,4,1,2]),XC_diff),4);
% CovX_X_repmat = sum(bsxfun(@times,permute(CovX_inv,[3,4,1,2]),X_repmat),4);

% cross validation
cv_fold=(1:op.cvfold);
cv_split=floor((0:op.samples-1)*op.cvfold./op.samples)+1;
cv_index=cv_split(randperm(op.samples));

b2=floor(op.bnum/2);

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
                psi_train=CovX_XC_diff(:,cv_index~=kk,dd).*GauKer(:,cv_index~=kk);  
                psi_test=CovX_XC_diff(:,cv_index==kk,dd).*GauKer(:,cv_index==kk); 
%                 phi_train=(CovX_XC_diff(:,cv_index~=kk,dd).*XC_diff(:,cv_index~=kk,dd)/sigma_tmp^2-CovX_inv(dd,dd)).*GauKer(:,cv_index~=kk);
%                 phi_test=(CovX_XC_diff(:,cv_index==kk,dd).*XC_diff(:,cv_index==kk,dd)/sigma_tmp^2-CovX_inv(dd,dd)).*GauKer(:,cv_index==kk);
                phi_train=(CovX_XC_diff(:,cv_index~=kk,dd).^2/sigma_tmp^2-CovX_inv(dd,dd)).*GauKer(:,cv_index~=kk);
                phi_test=(CovX_XC_diff(:,cv_index==kk,dd).^2/sigma_tmp^2-CovX_inv(dd,dd)).*GauKer(:,cv_index==kk);
            elseif op.bfunc == 2
                psi_train=CovX_XC_diff(:,cv_index~=kk,dd).*GauKer(:,cv_index~=kk);  
                psi_test=CovX_XC_diff(:,cv_index==kk,dd).*GauKer(:,cv_index==kk); 
%                 phi_train=(CovX_XC_diff(:,cv_index~=kk,dd).*XC_diff(:,cv_index~=kk,dd)/sigma_tmp^2-CovX_inv(dd,dd)).*GauKer(:,cv_index~=kk);
%                 phi_test=(CovX_XC_diff(:,cv_index==kk,dd).*XC_diff(:,cv_index==kk,dd)/sigma_tmp^2-CovX_inv(dd,dd)).*GauKer(:,cv_index==kk);
                phi_train=(CovX_XC_diff(:,cv_index~=kk,dd).*XC_diff(:,cv_index~=kk,dd)/sigma_tmp^2-CovX_inv(dd,dd)).*GauKer(:,cv_index~=kk);
                phi_test=(CovX_XC_diff(:,cv_index==kk,dd).*XC_diff(:,cv_index==kk,dd)/sigma_tmp^2-CovX_inv(dd,dd)).*GauKer(:,cv_index==kk);
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
%                 K_train_lam = K_train+lambda_tmp*diag(CovX(dd,:));
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
    fprintf('g: dd=%g, sigma=%g(id:%g), lambda=%g(id:%g)\n',dd,hparams.sigma(dd),hparams.sigma_id(dd),hparams.lambda(dd),hparams.lambda_id(dd));
end % dd

clear('psi_train','psi_test','phi_train','phi_test','K_train','K_test','h_train','h_test','K_train_lam','term1','term2','score_cv','thetah','sigma_tmp','lambda_tmp');
clear('cv_hold','cv_split','cv_index');

%% compute theta
theta=zeros(op.bnum,op.dim);
for dd=1:op.dim
    GauKer=exp(-XC_dist/(2*hparams.sigma(dd)^2));

    if op.bfunc == 1
        psi=CovX_XC_diff(:,:,dd).*GauKer;  
%         phi=(CovX_XC_diff(:,:,dd).*XC_diff(:,:,dd)/hparams.sigma(dd)^2-CovX_inv(dd,dd)).*GauKer;
        phi=(CovX_XC_diff(:,:,dd).^2/hparams.sigma(dd)^2-CovX_inv(dd,dd)).*GauKer;
    elseif op.bfunc == 2
        psi=CovX_XC_diff(:,:,dd).*GauKer;  
%         phi=(CovX_XC_diff(:,:,dd).*XC_diff(:,:,dd)/hparams.sigma(dd)^2-CovX_inv(dd,dd)).*GauKer;
        phi=(CovX_XC_diff(:,:,dd).*XC_diff(:,:,dd)/hparams.sigma(dd)^2-CovX_inv(dd,dd)).*GauKer;
    else
        psi=GauKer;
        phi=(XC_diff(:,:,dd)/hparams.sigma(dd)^2).*GauKer;
    end
    K=psi*psi'/size(psi,2);
    h=mean(phi,2);
    
    theta(:,dd)=linsolve(K+hparams.lambda(dd)*eye(size(K)),-h);
%     theta(:,dd)=linsolve(K+hparams.lambda(dd)*diag(CovX(dd,:)),-h);
end

clear('psi','phi','K','h','GauKer','dd');

%% compute grad 
GauKer3D=exp(-bsxfun(@rdivide,XC_dist,2*permute(hparams.sigma.^2,[2,3,1]))); % b*n*d

if op.bfunc == 1
    psi_g=CovX_XC_diff.*GauKer3D;
%     phi_tmp = bsxfun(@rdivide,XC_diff,permute(hparams.sigma,[2,3,1])); % b*n*d
%     phi_g = bsxfun(@times,bsxfun(@minus,bsxfun(@times,phi_tmp,permute(phi_tmp,[1,2,4,3])),permute(eye(op.dim),[3,4,1,2])),GauKer3D);%b*n*d*d
    phi_tmp = bsxfun(@rdivide,CovX_XC_diff,permute(hparams.sigma.^2,[2,3,1])); % b*n*d
    phi_g = bsxfun(@times,bsxfun(@minus,bsxfun(@times,phi_tmp,permute(CovX_XC_diff,[1,2,4,3])),permute(diag(diag(CovX_inv)),[3,4,1,2])),GauKer3D);%b*n*d*d %3‚Â–Ú‚ªj
elseif op.bfunc == 2
     psi_g=CovX_XC_diff.*GauKer3D;
%     phi_tmp = bsxfun(@rdivide,XC_diff,permute(hparams.sigma,[2,3,1])); % b*n*d
%     phi_g = bsxfun(@times,bsxfun(@minus,bsxfun(@times,phi_tmp,permute(phi_tmp,[1,2,4,3])),permute(eye(op.dim),[3,4,1,2])),GauKer3D);%b*n*d*d
    phi_tmp = bsxfun(@rdivide,CovX_XC_diff,permute(hparams.sigma.^2,[2,3,1])); % b*n*d
    phi_g = bsxfun(@times,bsxfun(@minus,bsxfun(@times,phi_tmp,permute(XC_diff,[1,2,4,3])),permute(diag(diag(CovX_inv)),[3,4,1,2])),GauKer3D);%b*n*d*d %3‚Â–Ú‚ªj
    
else
    psi_g=GauKer3D; %b*n*d
    phi_tmp=bsxfun(@times,permute(hparams.sigma.^2,[2,3,1]),GauKer3D);
    phi_g = bsxfun(@times,phi_tmp,permute(XC_diff,[1,2,4,3]));
end

clear('GauKer3D','phi_tmp');
clear('XC_diff','XC_dist')
 
g=permute(sum(bsxfun(@times,theta,permute(psi_g,[1,3,2])),1),[2,3,1]);
% nabla_g=permute(sum(bsxfun(@times,theta,permute(phi_g,[1,3,2])),1),[2,3,1]);
nabla_g = permute(sum(bsxfun(@times,theta,permute(phi_g,[1,3,4,2])),1),[2,3,4,1]);%b*d, b*d*d*n % d*d*n 1‚Â–Ú‚ªj
