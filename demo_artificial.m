function [Errs,exop] = demo_artificial(data_id,r_id,run_id)
%demo_r
% ノイズの共分散行列の条件数を変えた時の人工データ実験のデモ
% Input
%   data_id: 
%       1: Independent Gaussian Mixture
%       2: Dependent Super-Gaussian
%       3: Dependent Uniform
%       4: Dependent Super- and Sub-Gaussian
%   r_id:
%       r_id = 0,1,...,10
%       set r as r=0.1*r_id
%   run_id:
%       set a seed for random number
% Output
%   Errs: subspace estimation error of each algorithms
%   exop: options(folder names)

narginchk(2,3);

% set about random numbers
if ~exist('run_id','var') || isempty(run_id)
    run_id = 1;
end
exop.rand_id = run_id;
s = RandStream('mt19937ar','Seed',exop.rand_id);
RandStream.setGlobalStream(s); 

% set about save folders
exop.exname = 'artificial';
exop.datafolder = sprintf('%s/data%d',exop.exname,data_id);
mkdir(exop.datafolder);
exop.ex_id = sprintf('%s/r_id%d',exop.datafolder,r_id);
mkdir(exop.ex_id);

%% set options

op.dx= 10; % the number of data dimension
op.dz = 2; % the number of the dimension of non-Gaussian index subspace
op.bnum = 100; % the number of basis functions
op.bfunc = 1; % a kind of basis functions
op.n = 2000; % number of samples

op_LSNGCA = op; % options for LSNGCA
op_WFLSNGCA = op; % options for LSNGCA

op_LSNGCA.sigma_list = logspace(-1,1,10);
op_LSNGCA.lambda_list = logspace(-5,1,10);
op_LSNGCA.bfunc = 1;

op_WFLSNGCA.sigma_list = logspace(-1,1,10);
op_WFLSNGCA.sigma_list_v = logspace(-1,1,10);
op_WFLSNGCA.lambda_list = logspace(-5,1,10);
op_WFLSNGCA.lambda_list_v = logspace(-5,1,10);
op_WFLSNGCA.bfunc = 1;

%% gererate Signal
switch data_id
    case 1
        % Independent Gaussian Mixture
        S=zeros(op.dz,op.n);
        for ii=1:op.dz
            myu = [3;-3];
            Smat = cat(3,1,1);
            rp=0.5; p=[rp,1-rp];
            GMMobj = gmdistribution(myu,Smat,p);
            S(ii,:)=random(GMMobj,op.n)';%+randn(1,op.n)/4;
        end
    case 2
        % Dependent Super-Gaussian
        f = @(x) exp(-norm(x));
        S = slicesample([0;0],op.n,'pdf',f,'burnin',50000)';
    case 3
        % Dependent Uniform
        rr = sqrt(rand(1,op.n));
        theta = 2*pi*rand(1,op.n);
        S = [rr.*cos(theta);rr.*sin(theta)];
    case 4
        % Dependent Super- and Sub-Gaussian
        S1 = rand([1,op.n]);
        S2 = laprnd(1,op.n,1);
        S1(abs(S2)>log(2)) = S1(abs(S2)>log(2))-1;
        S = [S1;S2];
%     case 5
%         % Independent Super-Gaussian
%         f = @(x) exp(-norm(x));
%         S = [slicesample(0,op.n,'pdf',f,'burnin',50000)';slicesample(0,op.n,'pdf',f,'burnin',50000)'];    
%     case 6
%         % Indepemdemt Uniform
%         S = rand([2,op.n]);
%     case 7
%         % Independent Super- and Sub-Gaussian
%         f = @(x) exp(-norm(x)^4);
%         S1 = slicesample(0,op.n,'pdf',f,'burnin',50000)';
%         S2 = laprnd(1,op.n,1);
%         S = [S1;S2];              
%     case 8
%         % Independent Sub-Gaussian
%         f = @(x) exp(-norm(x)^4);
%         S = [slicesample(0,op.n,'pdf',f,'burnin',50000)';slicesample(0,op.n,'pdf',f,'burnin',50000)'];
%     case 9
%         % Dependent Sub-Gaussian
%         f = @(x) exp(-norm(x)^4);
%         S = slicesample([0;0],op.n,'pdf',f,'burnin',50000)'; 
end

% set r
r=r_assign(r_id);

% mix signal and noise
X = addnoise(S,r,op);

% centering & normalizing
X=bsxfun(@minus,X,mean(X,2));
X=bsxfun(@rdivide,X,std(X,[],2));

% calculate the number of condition
CovX = X*X'/op.n;
Ncond = cond(CovX/op.n)

%% do NGCA(MIPP)
addpath files_NGCA
ticID = tic();
[ngspace_NGCA,~,~]=NGCA(X,[]);
ctimes.NGCA = toc(ticID);

%% do LSNGCA
addpath files_LSNGCA
ticID = tic();
[ngspace_LSNGCA,~,~,hparams.LSNGCA]=LSNGCA(X,op_LSNGCA.dz,op_LSNGCA);
ctimes.LSNGCA = toc(ticID);

%% do SNGCA
%addpath newrelax
%maxIter = 1000;

%ticID = tic();
%P_SNGCA=mySNGCA(X,op.dz,maxIter);
%B_SNGCA = P_SNGCA;
%ngspace_SNGCA = B_SNGCA*inv(B_SNGCA'*B_SNGCA)^0.5;
%ctimes.SNGCA = toc(ticID);

%% do WF-LSNGCA
addpath files_WFLSNGCA
C = setKcenter(X,op_WFLSNGCA.bnum);

ticID = tic();
[ngspace_WF_LSNGCA,~,~,hparams.WF_LSNGCA]=WF_LSNGCA(X,op_WFLSNGCA.dz,op_WFLSNGCA,C);
ctimes.WF_LSNGCA = toc(ticID);
% ctimes.WF_LSNGCA = toc(ticID)+time_LSLDG;

%% evaluation
%

Errs.NGCA = 1-mean(sum(ngspace_NGCA(1:2,:).^2,1),2);
Errs.LSNGCA = 1-mean(sum(ngspace_LSNGCA(1:2,:).^2,1),2);
% Errs.SNGCA = 1-mean(sum(ngspace_SNGCA(1:2,:).^2,1),2);
Errs.WF_LSNGCA = 1-mean(sum(ngspace_WF_LSNGCA(1:2,:).^2,1),2);


clear('C','X','ticID','S','S1','S2','g','nabla_g','f','v');
save(sprintf('%s/%d.mat',exop.ex_id,exop.rand_id));

end

