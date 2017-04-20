function [exop,MissClassRate] = demo_bench(data_id,run_id,dx)
%DEMO_BENCH
%   ベンチマークデータを用いて，SVMを用いた分類精度を比較する
% Input:
%   data_id:
%   用いるベンチマークデータの種類を決めるindex．ClassData関数を用いて，訓練データ，テストデータ，各ラベルを取り出す．
%   run_id: set a seed for random number
%
%

exop.rand_id = run_id;
s = RandStream('mt19937ar','Seed',exop.rand_id);
RandStream.setGlobalStream(s);

addpath('~/old/workspace/libsvm-3.21/matlab') % add the pass the directly 'matlab' in libsvm directly.

[X_train,Y_train,X_test,Y_test,n_train,n_test,dataname]=ClassData(data_id,run_id);
mkdir('bench')
exop.datafolder = sprintf('bench/%s/d%d',dataname,dx);
mkdir(exop.datafolder)

%% set options
op.dx = dx;
op.n = n_train;
op.dz = size(X_train,1);
op.bnum = min(100,op.n);
op.bfunc = 1;

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

clear('X','Y','X1','X2','idx1','idx2');

%% add noise
d_noise = op.dx-op.dz;
% d_noise =-1;
if d_noise>0
    mu_noise = zeros(1,d_noise);
    base = ones(1,d_noise);
    Smat_noise_tmp = diag(base);
     
    Smat_noise = cat(3,Smat_noise_tmp);
    p_noise=1;
    GMMobj_noise = gmdistribution(mu_noise,Smat_noise,p_noise);
    N=random(GMMobj_noise,op.n+n_test)';
    
    N = bsxfun(@minus,N,mean(N,2));
    N = bsxfun(@rdivide,N,std(N,[],2));

    X_train = [X_train;N(:,1:op.n)];
    X_test = [X_test;N(:,op.n+1:op.n+n_test)];
end

mean_X_train = mean(X_train,2);
X_train = bsxfun(@minus,X_train,mean_X_train);
X_test = bsxfun(@minus,X_test,mean_X_train);

condition_number=cond(X_train*X_train'/op.n);
fprintf('condition number: %f\n',condition_number);

%% do PCA

ticID = tic();
B_PCA = pca(X_train');
ctime.PCA = toc(ticID);
P.PCA = B_PCA(:,1:op.dz)';
% P_PCA = B_PCA(1:dz,:);
Ztrain.PCA = P.PCA*X_train;
%
%% do NGCA(MIPP)
addpath files_NGCA
parameters.nbng = op.dz;
ticID = tic();
[~,Ztrain.NGCA,P.NGCA]=NGCA(X_train,parameters);
% [B_NGCA,Ztrain_NGCA]=NGCA(Xtrain,parameters);
% P_NGCA = B_NGCA';
ctime.NGCA = toc(ticID);
% 
%% do LSNGCA
addpath files_LSNGCA
ticID = tic();
[~,Ztrain.LSNGCA,P.LSNGCA,hparams.LSNGCA]=LSNGCA(X_train,op_LSNGCA.dz,op_LSNGCA);
ctime.LSNGCA = toc(ticID);

%% do WF-LSNGCA
addpath files_WFLSNGCA
C = setKcenter(X_train,op.bnum);

%     ticID = tic();
%     [~,Ztrain.WFLSNGCA_naive,P.WFLSNGCA_naive,hparams.WF_LSNGCA_naive,g,nabla_g,~,time_LSLDG]=WF_LSNGCA_naive(X_train,op.dz,op,C,MedDim);
%     ctime.WFLSNGCA_naive = toc(ticID);

ticID = tic();
[~,Ztrain.WFLSNGCA,P.WFLSNGCA,hparams.WF_LSNGCA]=WF_LSNGCA(X_train,op_WFLSNGCA.dz,op_WFLSNGCA,C);
ctime.WFLSNGCA = toc(ticID);
%     ctime.WFLSNGCA = toc(ticID)+time_LSLDG;

clear('ticID','g','nabla_g');

%% evaluation

% Base
[MissClassRate.Base,accuracy.Base] = eval_svm(X_train',Y_train',X_test',Y_test');

clear('X_train')

% PCA
Ztest.PCA = P.PCA*X_test;
[MissClassRate.PCA,accuracy.PCA] = eval_svm(Ztrain.PCA',Y_train',Ztest.PCA',Y_test');


%NGCA
Ztest.NGCA = P.NGCA*X_test;
[MissClassRate.NGCA,accuracy.NGCA] = eval_svm(Ztrain.NGCA',Y_train',Ztest.NGCA',Y_test');

%LSNGCA
Ztest.LSNGCA = P.LSNGCA*X_test;
[MissClassRate.LSNGCA,accuracy.LSNGCA] = eval_svm(Ztrain.LSNGCA',Y_train',Ztest.LSNGCA',Y_test');

%WFLSNGCA-naive
% Ztest.WFLSNGCA_naive = P.WFLSNGCA_naive*X_test;
% model.WFLSNGCA_naive = svmtrain(double(Y_train)',Ztrain.WFLSNGCA_naive');
% [~,accuracy.WFLSNGCA_naive,~] = svmpredict(double(Y_test'),Ztest.WFLSNGCA_naive',model.WFLSNGCA_naive);
% MissClassRate.WFLSNGCA_naive = 1-accuracy.WFLSNGCA_naive(1)/100;

%WFLSNGCA
Ztest.WFLSNGCA = P.WFLSNGCA*X_test;
[MissClassRate.WFLSNGCA,accuracy.WFLSNGCA] = eval_svm(Ztrain.WFLSNGCA',Y_train',Ztest.WFLSNGCA',Y_test');

%% save data
clear('fig');
clear('C','X','ticID','S','S1','S2','g','nabla_g','X_train','X_test','Y_train','Y_test','Ztrain','Ztest','N','Smat_noise','R');
clear('model','accuracy','MedDim','parameters','B_PCA')

save(sprintf('%s/%d.mat',exop.datafolder,exop.rand_id));
end

function [MissClassRate,accuracy] = eval_svm(Xtrain,Ytrain,Xtest,Ytest)
  bestcv = 0;

  for log2c = -1:3
    for log2g = -4:1
      cv_cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
      cv = svmtrain(double(Ytrain),Xtrain,cv_cmd);
      if cv >= bestcv
        bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
      end
    end
  end

  fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg)];
  %model = svmtrain(double(Ytrain),Xtrain,cmd);
  model = svmtrain(double(Ytrain),Xtrain);
  [~,accuracy,~] = svmpredict(double(Ytest),Xtest,model);
  MissClassRate = 1-accuracy(1)/100;
end
