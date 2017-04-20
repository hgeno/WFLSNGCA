function [ngmatrix,projdata,projmat,hparams,g,v]=WF_LSNGCA(X,dz,op,C,MedDims,g,nabla_g)
% WF-LSNGCA perform Whitening-Free LSNGCA
% Inputs
%   X       : data matrix (op.dim*op.samples).
%   dz      : dimension of projected data.
%   bnum    : number of kernel centers.
% Outputs
%   C       : kernel centers (op.dim*op.bnum).
%   MedDims :  for candidates of band-width (op.dim*1).
%   g       : log-density gradients of data samples (op.dim*op.samples).
%   nabla_g : derivatives of log-density gradients of data samples (op.dim*oop.dim*p.samples).
%

disp('WF-LSNGCA');
narginchk(3,7);

%% set variables
[op.dim,op.samples]=size(X);

if ~isfield(op,'bnum')
    op.bnum=min(op.samples,100);
end

if ~isfield(op,'sigma_list')
    op.sigma_list=logspace(-1,1,10);
end

if ~isfield(op,'lambda_list')
    op.lambda_list=logspace(-5,1,10);
end

if ~exist('C','var') || isempty(C)
    C = setKcenter(X,op.bnum);
end

%% main

% estimate the log-density gradient and calculate the derivatives of them
if ~exist('g','var') || isempty(g) || ~exist('nabla_g','var') || isempty(nabla_g)
     [g,nabla_g,~,hparams.hparams_g] = LSLDG_for_WFLSNGCA(X,op,C);
end

% estimate the non-Gaussian index vectors
[v,~,hparams.ngiv] = LSNGIV(X,nabla_g,op,C);

% perform eigen-decomposition of covariance matrix of v
Sigma = v*v'/op.samples;
[E,D]=eig(Sigma); 

% large 
dd=diag(D);
[~,ind]=sort(dd,'descend');
B=E(:,ind(1:dz));

ngmatrix=B*inv(B'*B)^0.5;
projdata=B'*X;
projmat=B'; % projecting matrix 
