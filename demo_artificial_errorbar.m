function []=demo_artificial_errorbar(data_id,rlist,exop)

demo_dir = exop.datafolder;

error_plot_dir = sprintf('%s/error_plot',exop.exname);
mkdir(error_plot_dir);
ctime_plot_dir = sprintf('%s/ctime_plot',exop.exname);
mkdir(ctime_plot_dir);

rlist = rlist';
Nr = size(rlist,1);

Nconds_means = zeros(Nr,1);
Nconds_stds = zeros(Nr,1);

Errs_NGCA_means = zeros(Nr,1);
Errs_LSNGCA_means = zeros(Nr,1);
% Errs_SNGCA_means = zeros(Nr,1);
Errs_WF_LSNGCA_means = zeros(Nr,1);

Errs_NGCA_stds = zeros(Nr,1);
Errs_LSNGCA_stds = zeros(Nr,1);
% Errs_SNGCA_stds = zeros(Nr,1);
Errs_WF_LSNGCA_stds = zeros(Nr,1);

ctimes_NGCA_means = zeros(Nr,1);
ctimes_LSNGCA_means = zeros(Nr,1);
% ctimes_SNGCA_means = zeros(Nr,1);
ctimes_WF_LSNGCA_means = zeros(Nr,1);

ctimes_NGCA_stds = zeros(Nr,1);
ctimes_LSNGCA_stds = zeros(Nr,1);
% ctimes_SNGCA_stds = zeros(Nr,1);
ctimes_WF_LSNGCA_stds = zeros(Nr,1);

for ri = 1:Nr
    r_id2 = rlist(ri);
    result_dir = sprintf('%s/r_id%d',demo_dir,r_id2);
    load(sprintf('%s/result_data.mat',result_dir));

    Nconds_means(ri) = Nconds_mean;
    Nconds_stds(ri) = Nconds_std;
    
    Errs_NGCA_means(ri) = Errs_NGCA_mean;
    Errs_LSNGCA_means(ri) = Errs_LSNGCA_mean;
%     Errs_SNGCA_means(ri) = Errs_SNGCA_mean;
    Errs_WF_LSNGCA_means(ri) = Errs_WF_LSNGCA_mean;

    Errs_NGCA_stds(ri) = Errs_NGCA_std;
    Errs_LSNGCA_stds(ri) = Errs_LSNGCA_std;
%     Errs_SNGCA_stds(ri) = Errs_SNGCA_std;
    Errs_WF_LSNGCA_stds(ri) = Errs_WF_LSNGCA_std;

    ctimes_NGCA_means(ri) = ctimes_NGCA_mean;
    ctimes_LSNGCA_means(ri) = ctimes_LSNGCA_mean;
%     ctimes_SNGCA_means(ri) = ctimes_SNGCA_mean;
    ctimes_WF_LSNGCA_means(ri) = ctimes_WF_LSNGCA_mean;

    ctimes_NGCA_stds(ri) = ctimes_NGCA_std;
    ctimes_LSNGCA_stds(ri) = ctimes_LSNGCA_std;
%     ctimes_SNGCA_stds(ri) = ctimes_SNGCA_std;
    ctimes_WF_LSNGCA_stds(ri) = ctimes_WF_LSNGCA_std;
   
    rlist(ri) = r_assign(r_id2); % rlist に rの値を代入（index値に上書き）
end

labels = {'MIPP','LSNGCA','WF-LSNGCA'};
fig4 = figure(4);clf;hold on;

errorbar(rlist,Errs_NGCA_means,Errs_NGCA_stds,'k:','LineWidth',5);
errorbar(rlist,Errs_LSNGCA_means,Errs_LSNGCA_stds,'b--','LineWidth',5);
errorbar(rlist,Errs_WF_LSNGCA_means,Errs_WF_LSNGCA_stds,'r-','LineWidth',5);
% LC:with Local Covariance
ax = gca;
ax.FontSize = 24;
legend(labels,'Location','northwest','FontSize',20,'FontWeight','bold');
xlim([-0.05,0.85]);
y_max = max([Errs_NGCA_means+Errs_NGCA_stds;Errs_LSNGCA_means+Errs_LSNGCA_stds;Errs_WF_LSNGCA_means+Errs_WF_LSNGCA_stds]);
ylim([0,min(1,y_max)]);
xlabel('r for Noise Covariances','FontSize',28,'FontWeight','bold');
ylabel('Subspace Estimation Error','FontSize',28,'FontWeight','bold');

saveimg(fig4, sprintf('%s/error_plot_r-data%d',error_plot_dir,data_id));

% labels = {'Condition Number'};
fig6 = figure(6);clf;hold on;
errorbar(rlist,Nconds_means,Nconds_stds,'r-','LineWidth',5);
% LC:with Local Covariance
%legend(labels,'Location','northwest','FontSize',18,'FontWeight','bold');
ax = gca;
ax.FontSize = 24;
xlim([-0.05,0.85]);
xlabel('r for Noise Covariances','FontSize',28,'FontWeight','bold');
ylabel('Condition Number','FontSize',28,'FontWeight','bold');

saveimg(fig6,sprintf('%s/condition_number-data%d',error_plot_dir,data_id));

end

