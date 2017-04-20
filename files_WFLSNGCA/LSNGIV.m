function [v,Theta_v,hparams,MedDim,C,cind]=LSNGIV(X,nabla_g,op,C,MedDim)
%
% Estimating Log-Density Gradients(multi-band)
% Input:
%   X : (sample) by (dim) matrix
%   nabla_g : derivatives of log-density gradients (op.dim*op.dim*op.samples)
%              (1éüå≥ñ⁄Ç∆ÇQéüå≥ñ⁄ÇÃèáî‘Ç…íçà”)
%   op: options
%   C : center points of kernels
%   MedDim : bases for band-width
% Output
%   v : estimated non-Gaussian index vectors (op.dim*op.samples)
%   Theta_v : learned parameters (op.bnum*op.dim)
%   hparams: selected hyper-parameters
%   C : center points of kernels
%   cind : index of data selected as the kernel senters
%   MedDim : bases for band-width

narginchk(2,5);

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

if ~exist('MedDim','var') || isempty(MedDim)
    MedDim=ones(op.dim,1);
end
%}
% MedDim=repmat(MedianDiff(X),[op.dim,1]);


% Difference to centers (size: bnum by samles by dim)
XC_diff=repmat(permute(C,[2,3,1]),[1,op.samples,1])...
    -repmat(permute(X,[3,2,1]),[op.bnum,1,1]);

% Distance from centers (size: bnum by samples)
XC_dist=sum(XC_diff.^2,3);

X_XC_diff = sum(bsxfun(@times,permute(X,[3,2,1]),XC_diff),3);
X_repmat = repmat(permute(X,[3,2,1]),[op.bnum,1,1]);%b*n*d
X2_repmat = repmat(diag(X'*X)',[op.bnum,1]);

% cross validation
cv_fold=(1:op.cvfold);
cv_split=floor((0:op.samples-1)*op.cvfold./op.samples)+1;
cv_index=cv_split(randperm(op.samples));

hparams.sigma_ggx=zeros(op.dim,1); hparams.lambda_ggx=zeros(op.dim,1);
hparams.sigma_ggx_id=zeros(op.dim,1); hparams.lambda_ggx_id=zeros(op.dim,1);
for dd=1:op.dim                
    score_cv_v=zeros(length(op.sigma_list_v),length(op.lambda_list_v),length(cv_fold));
%     one_dd = zeros(1,1,op.dim);
%     one_dd(:,:,dd) = 1;

    for sigma_index=1:length(op.sigma_list_v)
        sigma_tmp=MedDim(dd)*op.sigma_list_v(sigma_index);
%         sigma_tmp=op.sigma_list(sigma_index);
        GauKer=exp(-XC_dist/(2*sigma_tmp^2));
        
        for kk=cv_fold
            if op.bfunc == 2
                psi_train = (XC_diff(:,cv_index~=kk,dd).*X_XC_diff(:,cv_index~=kk)-X_repmat(:,cv_index~=kk,dd)*sigma_tmp^2).*GauKer(:,cv_index~=kk);
                psi_test = (XC_diff(:,cv_index==kk,dd).*X_XC_diff(:,cv_index==kk)-X_repmat(:,cv_index==kk,dd)*sigma_tmp^2).*GauKer(:,cv_index==kk); 
                phi_train = (XC_diff(:,cv_index~=kk,dd).^2-2*XC_diff(:,cv_index~=kk,dd).*X_repmat(:,cv_index~=kk,dd)-X_XC_diff(:,cv_index~=kk)-sigma_tmp^2+...
                             (X_XC_diff(:,cv_index~=kk).*XC_diff(:,cv_index~=kk,dd).^2)/sigma_tmp^2).*GauKer(:,cv_index~=kk);
                phi_test = (XC_diff(:,cv_index==kk,dd).^2-2*XC_diff(:,cv_index==kk,dd).*X_repmat(:,cv_index==kk,dd)-X_XC_diff(:,cv_index==kk)-sigma_tmp^2+...
                             (X_XC_diff(:,cv_index==kk).*XC_diff(:,cv_index==kk,dd).^2)/sigma_tmp^2).*GauKer(:,cv_index==kk);
%                 xi_train = (-2*X_XC_diff(:,cv_index~=kk)+X2_repmat(:,cv_index~=kk)+...
%                     ((2*X_XC_diff(:,cv_index~=kk)-X2_repmat(:,cv_index~=kk)).*XC_diff(:,cv_index~=kk,dd).^2-2*X_XC_diff(:,cv_index~=kk).*X_repmat(:,cv_index~=kk,dd).*XC_diff(:,cv_index~=kk,dd)-X_XC_diff(:,cv_index~=kk).^2)/sigma_tmp^2+ ...
%                      (X_XC_diff(:,cv_index~=kk).^2.*XC_diff(:,cv_index~=kk,dd).^2)/sigma_tmp^4 ...
%                  -4*X_repmat(:,cv_index~=kk,dd).*XC_diff(:,cv_index~=kk,dd)+2*X_repmat(:,cv_index~=kk,dd).^2-2*(X_XC_diff(:,cv_index~=kk).*X_repmat(:,cv_index~=kk,dd).*XC_diff(:,cv_index~=kk,dd))/sigma_tmp^2)...
%                  .*GauKer(:,cv_index~=kk);
%                 xi_test = (-2*X_XC_diff(:,cv_index==kk)+X2_repmat(:,cv_index==kk)+...
%                     ((2*X_XC_diff(:,cv_index==kk)-X2_repmat(:,cv_index==kk)).*XC_diff(:,cv_index==kk,dd).^2-2*X_XC_diff(:,cv_index==kk).*X_repmat(:,cv_index==kk,dd).*XC_diff(:,cv_index==kk,dd)-X_XC_diff(:,cv_index==kk).^2)/sigma_tmp^2+ ...
%                      (X_XC_diff(:,cv_index==kk).^2.*XC_diff(:,cv_index==kk,dd).^2)/sigma_tmp^4 ...
%                  -4*X_repmat(:,cv_index==kk,dd).*XC_diff(:,cv_index==kk,dd)+2*X_repmat(:,cv_index==kk,dd).^2-2*(X_XC_diff(:,cv_index==kk).*X_repmat(:,cv_index==kk,dd).*XC_diff(:,cv_index==kk,dd))/sigma_tmp^2)...
%                  .*GauKer(:,cv_index==kk);
            elseif op.bfunc == 1
                psi_train = XC_diff(:,cv_index~=kk,dd).*GauKer(:,cv_index~=kk);
                psi_test = XC_diff(:,cv_index==kk,dd).*GauKer(:,cv_index==kk);
                phi_train=(XC_diff(:,cv_index~=kk,dd).^2/sigma_tmp^2-1).*GauKer(:,cv_index~=kk);
                phi_test=(XC_diff(:,cv_index==kk,dd).^2/sigma_tmp^2-1).*GauKer(:,cv_index==kk);
%                 xi_train = ((X_XC_diff(:,cv_index~=kk)-2*X_repmat(:,cv_index~=kk,dd).*XC_diff(:,cv_index~=kk,dd))/sigma_tmp^2 ...
%                             -XC_diff(:,cv_index~=kk,dd).*X_XC_diff(:,cv_index~=kk)/sigma_tmp^4).*GauKer(:,cv_index~=kk);
%                 xi_test = ((X_XC_diff(:,cv_index==kk)-2*X_repmat(:,cv_index==kk,dd).*XC_diff(:,cv_index==kk,dd))/sigma_tmp^2 ...
%                              -XC_diff(:,cv_index==kk,dd).*X_XC_diff(:,cv_index==kk)/sigma_tmp^4).*GauKer(:,cv_index==kk);
            elseif op.bfunc == 3
                psi_train = bsxfun(@times,C(dd,:)',GauKer(:,cv_index~=kk));
                psi_test = bsxfun(@times,C(dd,:)',GauKer(:,cv_index==kk));
                phi_train = bsxfun(@times,XC_diff(:,cv_index~=kk,dd),C(dd,:)')/sigma_tmp^2.*GauKer(:,cv_index~=kk);
                phi_test = bsxfun(@times,XC_diff(:,cv_index==kk,dd),C(dd,:)')/sigma_tmp^2.*GauKer(:,cv_index==kk);
%                 xi_train = ((X_XC_diff(:,cv_index~=kk)-2*X_repmat(:,cv_index~=kk,dd).*XC_diff(:,cv_index~=kk,dd))/sigma_tmp^2 ...
%                             -XC_diff(:,cv_index~=kk,dd).*X_XC_diff(:,cv_index~=kk)/sigma_tmp^4).*GauKer(:,cv_index~=kk);
%                 xi_test = ((X_XC_diff(:,cv_index==kk)-2*X_repmat(:,cv_index==kk,dd).*XC_diff(:,cv_index==kk,dd))/sigma_tmp^2 ...
%                              -XC_diff(:,cv_index==kk,dd).*X_XC_diff(:,cv_index==kk)/sigma_tmp^4).*GauKer(:,cv_index==kk);
            else 
                psi_train = GauKer(:,cv_index~=kk); 
                psi_test=GauKer(:,cv_index==kk);
                phi_train=(XC_diff(:,cv_index~=kk,dd)/sigma_tmp^2).*GauKer(:,cv_index~=kk);
                phi_test=(XC_diff(:,cv_index==kk,dd)/sigma_tmp^2).*GauKer(:,cv_index==kk);
%                 xi_train = (X_XC_diff(:,cv_index~=kk).*XC_diff(:,cv_index~=kk,dd)/sigma_tmp^4-X_repmat(:,cv_index~=kk,dd)/sigma_tmp^2).*GauKer(:,cv_index~=kk); %b*n
%                 xi_test = (X_XC_diff(:,cv_index==kk).*XC_diff(:,cv_index==kk,dd)/sigma_tmp^4-X_repmat(:,cv_index==kk,dd)/sigma_tmp^2).*GauKer(:,cv_index==kk); %b*n
            end

            K_train_v=psi_train*psi_train'/size(psi_train,2);
            K_test_v=psi_test*psi_test'/size(psi_test,2);
            h_train_v=mean(phi_train,2)+mean(bsxfun(@times,psi_train,(sum(permute(nabla_g(dd,:,cv_index~=kk),[2,3,1]).*X(:,cv_index~=kk),1))),2);             
            h_test_v=mean(phi_test,2)+mean(bsxfun(@times,psi_test,(sum(permute(nabla_g(dd,:,cv_index==kk),[2,3,1]).*X(:,cv_index==kk),1))),2);          
            
            for lambda_index=1:length(op.lambda_list_v)
                lambda_tmp=op.lambda_list_v(lambda_index); 
                K_train_v_lam = K_train_v+lambda_tmp*eye(op.bnum);
                if rank(K_train_v_lam) == op.bnum
                    thetah_v=linsolve(K_train_v_lam,-h_train_v);

                    term1_v=thetah_v'*K_test_v*thetah_v;
                    term2_v=thetah_v'*h_test_v;

                    score_cv_v(sigma_index,lambda_index,kk)=term1_v+2*term2_v;
                else
                    score_cv_v(sigma_index,lambda_index,kk) = Inf;
                end
            end %lambda                            
        end % kk
    end % sigma

    [score_cv_ggx_tmp,lambda_index_list_ggx]=min(mean(score_cv_v,3),[],2);
    if sum(score_cv_ggx_tmp(score_cv_ggx_tmp~=Inf))==0
        disp('NGIV:Please change hyper-parameter list');
%         exit;
    end
    [~,hparams.sigma_ggx_id(dd)]=min(score_cv_ggx_tmp);
    hparams.sigma_ggx(dd)=MedDim(dd)*op.sigma_list_v(hparams.sigma_ggx_id(dd));
%     sigma(dd)=op.sigma_list(sigma_index);
    hparams.lambda_ggx_id(dd) = lambda_index_list_ggx(hparams.sigma_ggx_id(dd));
    hparams.lambda_ggx(dd)=op.lambda_list_v(hparams.lambda_ggx_id(dd));

    fprintf('NGIV: dd=%g, sigma=%g(id=%d), lambda=%g(id=%d)\n',dd,hparams.sigma_ggx(dd),hparams.sigma_ggx_id(dd),hparams.lambda_ggx(dd),hparams.lambda_ggx_id(dd));
end % dd

clear sigma_tmp

%% compute Theta_ggx
Theta_v=zeros(op.bnum,op.dim);
for dd=1:op.dim
    GauKer_v=exp(-XC_dist/(2*hparams.sigma_ggx(dd)^2));
%     one_dd = zeros(1,op.dim);
%     one_dd(dd) = 1;
    
    if op.bfunc == 2
        psi_v = (XC_diff(:,:,dd).*X_XC_diff-X_repmat(:,:,dd)*hparams.sigma_ggx(dd)^2).*GauKer_v;
        phi_v = (XC_diff(:,:,dd).^2-2*XC_diff(:,:,dd).*X_repmat(:,:,dd)-X_XC_diff-hparams.sigma_ggx(dd)^2+...
                     (X_XC_diff.*XC_diff(:,:,dd).^2)/hparams.sigma_ggx(dd)^2).*GauKer_v;
%         xi_ggx = (-2*X_XC_diff+X2_repmat+...
%             ((2*X_XC_diff-X2_repmat).*XC_diff(:,:,dd).^2-2*X_XC_diff.*X_repmat(:,:,dd).*XC_diff(:,:,dd)-X_XC_diff.^2)/hparams.sigma_ggx(dd)^2+ ...
%              (X_XC_diff.^2.*XC_diff(:,:,dd).^2)/hparams.sigma_ggx(dd)^4 ...
%          -4*X_repmat(:,:,dd).*XC_diff(:,:,dd)+2*X_repmat(:,:,dd).^2-2*(X_XC_diff.*X_repmat(:,:,dd).*XC_diff(:,:,dd))/hparams.sigma_ggx(dd)^2)...
%          .*GauKer_ggx;
    elseif op.bfunc == 1
        psi_v = XC_diff(:,:,dd).*GauKer_v;
        phi_v=(XC_diff(:,:,dd).^2/hparams.sigma_ggx(dd)^2-1).*GauKer_v;
%         xi_ggx = ((X_XC_diff-2*X_repmat(:,:,dd).*XC_diff(:,:,dd))/hparams.sigma_ggx(dd)^2 ...
%                     -XC_diff(:,:,dd).*X_XC_diff/hparams.sigma_ggx(dd)^4).*GauKer_ggx;
    elseif op.bfunc == 3
        psi_v = bsxfun(@times,C(dd,:)',GauKer_v);
        phi_v = bsxfun(@times,XC_diff(:,:,dd),C(dd,:)')/hparams.sigma_ggx(dd)^2.*GauKer_v;
    else
        psi_v = GauKer_v;
        phi_v=(XC_diff(:,:,dd)./hparams.sigma_ggx(dd)^2).*GauKer_v;
%         xi_ggx = (X_XC_diff.*XC_diff(:,:,dd)/hparams.sigma_ggx(dd)^4-X_repmat(:,:,dd)/hparams.sigma_ggx(dd)^2).*GauKer_ggx; %b*n
%         xi_ggx=bsxfun(@times,bsxfun(@minus,permute(bsxfun(@times,XC_diff(:,:,dd),XC_diff),[1,3,2])/hparams.sigma_ggx(dd)^4,one_dd/hparams.sigma_ggx(dd)^2)...%b*d*n
%             ,permute(GauKer_ggx,[1,3,2]));%b*d*n
    end

    K_v=psi_v*psi_v'/size(psi_train,2);
    h_v=mean(phi_v,2)+mean(bsxfun(@times,psi_v,(sum(permute(nabla_g(dd,:,:),[2,3,1]).*X,1))),2);

    Theta_v(:,dd)=linsolve(K_v+hparams.lambda_ggx(dd)*eye(op.bnum),-h_v);
end

%% compute ggx
GaussKer3D = exp(-bsxfun(@rdivide,XC_dist,2*permute(hparams.sigma_ggx.^2,[2,3,1])));%b*n*d

if op.bfunc ==2
    psi_v3D = (XC_diff.*repmat(X_XC_diff,[1,1,op.dim])-bsxfun(@times,X_repmat,permute(hparams.sigma_ggx.^2,[2,3,1]))).*GaussKer3D; %b*n*d
elseif op.bfunc == 1
    psi_v3D = XC_diff.*GaussKer3D;
elseif op.bfunc == 3
    psi_v3D = bsxfun(@times,permute(C,[2,3,1]),GaussKer3D);
else
    psi_v3D=GaussKer3D;
end

v=permute(sum(bsxfun(@times,Theta_v,permute(psi_v3D,[1,3,2])),1),[2,3,1]);
