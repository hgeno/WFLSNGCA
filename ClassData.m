function [X_train,label_train,X_test,label_test,n_train,n_test,dataname]=ClassData(ii,seed)

rpath='datafiles/';

rng('default'); rng(seed);

if ii==1
    dataname = 'ijcnn1';
    [label,X]=libsvmread([rpath dataname]);

    n_train=2000; 
elseif ii==2
    dataname = 'SUSY';
    [label,X]=libsvmread([rpath dataname]);    
    label=sign(label-0.5);

    n_train=2000;
%     n_te=1000;
elseif ii==3
    dataname = 'shuttle.scale';
    [label,X]=libsvmread([rpath dataname]);    
  %1 4 5 label many      

    ll1=1; ll2=4;
    Y=[X(label==ll1,:);X(label==ll2,:)]; X=Y;
    tmp=[label(label==ll1);label(label==ll2)]; label=tmp;
    n_train = 2000;
elseif ii==4
    dataname = 'vehicle.scale';
    [label,X]=libsvmread([rpath dataname]);    
    
    tmp=zeros(size(X,2),1);
    tmp(label==1)=1;  tmp(label==3)=1;
    tmp(label==2)=-1;  tmp(label==4)=-1;
    label=tmp;

    n_train=200;
elseif ii==5
    dataname = 'svmguide3';
    [label,X]=libsvmread([rpath dataname]);    
    X(:,end)=[];
    
    n_train=200;
elseif ii==6
    dataname = 'svmguide1';
    [label,X]=libsvmread([rpath dataname]);    
    X(:,end)=[];
    [label_tmp,X_tmp]=libsvmread([rpath 'svmguide1.t']);    
    X_tmp(:,end)=[];
    X = [X;X_tmp];
    label = [label;label_tmp];
    
    n_train=2000;
else    
    error('Check the data number');
end

X=full(X');

n_test = n_train;

%shuffle
X1 = X(:,label==1);
rind1=randperm(size(X1,2));
X1=X1(:,rind1);

X2 = X(:,label~=1);
rind2=randperm(size(X2,2));
X2=X2(:,rind2);

%create train data
X_train=[X1(:,rind1(1:n_train/2)),X2(:,rind2(1:n_train/2))];
rind1(1:n_train/2) = [];
rind2(1:n_train/2) = [];
label_train=[ones(1,n_train/2) -ones(1,n_train/2)];

rtrain = randperm(size(X_train,2));
X_train = X_train(:,rtrain);
label_train = label_train(:,rtrain);

X_train_mean = mean(X,2);
X_train_std = std(X,[],2);
X_train = bsxfun(@minus,X_train,X_train_mean);
X_train = bsxfun(@rdivide,X_train,X_train_std);

%create test data
X_test=[X1(:,rind1(1:n_test/2)),X2(:,rind2(1:n_test/2))];
label_test = [ones(1,n_test/2) -ones(1,n_test/2)];

rtest = randperm(size(X_test,2));
X_test = X_test(:,rtest);
label_test = label_test(:,rtest);
X_test = bsxfun(@minus,X_test,X_train_mean);
X_test = bsxfun(@rdivide,X_test,X_train_std);

