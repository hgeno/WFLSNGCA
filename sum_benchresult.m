function [] = sum_benchresult(dir_result,Nruns)
%DEMO_LSNGCA Ç±ÇÃä÷êîÇÃäTóvÇÇ±Ç±Ç…ãLèq

MissClassRates.Base = zeros(1,Nruns); 
MissClassRates.PCA = zeros(1,Nruns); 
MissClassRates.NGCA = zeros(1,Nruns); 
MissClassRates.LSNGCA = zeros(1,Nruns); 
MissClassRates.WFLSNGCA = zeros(1,Nruns);

%ctimes.Base = zeros(1,Nruns); 
ctimes.PCA = zeros(1,Nruns); 
ctimes.NGCA = zeros(1,Nruns); 
ctimes.LSNGCA = zeros(1,Nruns); 
ctimes.WFLSNGCA = zeros(1,Nruns);

conditionNumbers = zeros(1,Nruns);

for run_id = 1:Nruns
   
    load(sprintf('%s/%d.mat',dir_result,run_id));
    
    MissClassRates.Base(run_id) = MissClassRate.Base;
    MissClassRates.PCA(run_id) =MissClassRate.PCA;
    MissClassRates.NGCA(run_id) =MissClassRate.NGCA;
    MissClassRates.LSNGCA(run_id) =MissClassRate.LSNGCA;
%     MissClassRates.WFLSNGCA_naive(run_id) =MissClassRate.WFLSNGCA_naive;
    MissClassRates.WFLSNGCA(run_id) =MissClassRate.WFLSNGCA;
    
%     ctimes.Base(run_id) = ctime.Base;
    ctimes.PCA(run_id) =ctime.PCA;
    ctimes.NGCA(run_id) =ctime.NGCA;
    ctimes.LSNGCA(run_id) =ctime.LSNGCA;
%     ctimes.WFLSNGCA_naive(run_id) =ctime.WFLSNGCA_naive;
    ctimes.WFLSNGCA(run_id) =ctime.WFLSNGCA;

    conditionNumbers(run_id) = condition_number;
end

MissClassRates.LSNGCA

Mean_MissClassRates.Base = mean(MissClassRates.Base,2);
Mean_MissClassRates.PCA = mean(MissClassRates.PCA,2);
Mean_MissClassRates.NGCA = mean(MissClassRates.NGCA,2);
Mean_MissClassRates.LSNGCA = mean(MissClassRates.LSNGCA,2);
% Mean_MissClassRates.WFLSNGCA_naive = mean(MissClassRates.WFLSNGCA_naive,2);
Mean_MissClassRates.WFLSNGCA = mean(MissClassRates.WFLSNGCA,2);

std_MissClassRates.Base = std(MissClassRates.Base,[],2);
std_MissClassRates.PCA = std(MissClassRates.PCA,[],2);
std_MissClassRates.NGCA = std(MissClassRates.NGCA,[],2);
std_MissClassRates.LSNGCA = std(MissClassRates.LSNGCA,[],2);
% std_MissClassRates.WFLSNGCA_naive = std(MissClassRates.WFLSNGCA_naive,[],2);
std_MissClassRates.WFLSNGCA = std(MissClassRates.WFLSNGCA,[],2);

% Mean_ctimes.Base = mean(MissClassRates.Base,2);
Mean_ctimes.PCA = mean(ctimes.PCA,2);
Mean_ctimes.NGCA = mean(ctimes.NGCA,2);
Mean_ctimes.LSNGCA = mean(ctimes.LSNGCA,2);
% Mean_ctimes.WFLSNGCA_naive = mean(ctimes.WFLSNGCA_naive,2);
Mean_ctimes.WFLSNGCA = mean(ctimes.WFLSNGCA,2);

% std_ctimes.Base = std(ctimes.Base,[],2);
std_ctimes.PCA = std(ctimes.PCA,[],2);
std_ctimes.NGCA = std(ctimes.NGCA,[],2);
std_ctimes.LSNGCA = std(ctimes.LSNGCA,[],2);
% std_ctimes.WFLSNGCA_naive = std(ctimes.WFLSNGCA_naive,[],2);
std_ctimes.WFLSNGCA = std(ctimes.WFLSNGCA,[],2);

%% t-test

% Mean_MissClassRate_list =[Mean_MissClassRates.Base Mean_MissClassRates.PCA Mean_MissClassRates.NGCA Mean_MissClassRates.LSNGCA Mean_MissClassRates.WFLSNGCA_naive Mean_MissClassRates.WFLSNGCA];

MissClassRate_list = [MissClassRates.Base;MissClassRates.PCA;MissClassRates.NGCA;MissClassRates.LSNGCA;MissClassRates.WFLSNGCA];
Mean_MissClassRate_list =[Mean_MissClassRates.Base Mean_MissClassRates.PCA Mean_MissClassRates.NGCA Mean_MissClassRates.LSNGCA Mean_MissClassRates.WFLSNGCA];
[~,min_alg_id] = min(Mean_MissClassRate_list);

Nalg= size(Mean_MissClassRate_list,2);
T = ones(1,Nalg);
% Mat_ttest = zeros(Nalg,Nalg);
for alg_id = 1:Nalg
        T(alg_id)=ttest2(MissClassRate_list(alg_id,:)',MissClassRate_list(min_alg_id,:)');
end

Mean_MissClassRates
std_MissClassRates
T
meanConditionNumber = mean(conditionNumbers,2)
stdConditionNumber = std(conditionNumbers,[],2)

end

