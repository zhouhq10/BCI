function call_fhpred_master
% function call_fhpred_master
% this function cycles through subjects and classes, generating aggregate
% stats, and makes figures
% kjm, 2015


%% subjects and classes

subjects = {...   
    'ja', ...
    'ca', ...
    'mv', ...
    'zt', ...
    'wc', ...
    'de', ...
    'fp', ...
};

cls_all= {'erp','bb','bth'};


%%
for k=1:length(subjects)
    disp(subjects{k})
%        mkdir([cd '/data/' subjects{k}])
%        mkdir([cd '/figs/' subjects{k}])
       fhpred_cls_all{k}=fhpred_master(subjects{k});
end

%% continuous classfication

for k=1:length(subjects)
for m=1:3 % cycle through classes
    disp(cls_all{m})
    pct_capt(k,m)=size(fhpred_cls_all{k}{m}.evlink,1)/size(fhpred_cls_all{k}{m}.evtest,1);
    pct_err(k,m)=size(fhpred_cls_all{k}{m}.evfalse,1)/(size(fhpred_cls_all{k}{m}.evfalse,1)+size(fhpred_cls_all{k}{m}.evlink,1));
    timeerr_mean(k,m)=mean(abs(fhpred_cls_all{k}{m}.evlink(:,1)));
    timeerr_std(k,m)=std(abs(fhpred_cls_all{k}{m}.evlink(:,1)));
end
end




%%
figure,
subplot(3,1,1),kjm_errbar_grps(pct_capt.',NaN*0*disc_corr.',NaN*0*disc_corr.')
set(gca,'ylim',[.74 1]),box off,set(gca,'ytick',[.75:.05:1]),set(gca,'xtick',[]),title('Portion of stimuli predicted'),ylabel('Fraction')
set(gca,'ygrid','on')

subplot(3,1,2),kjm_errbar_grps(timeerr_mean.',timeerr_std.',timeerr_std.')
box off,set(gca,'xtick',[]),ylabel('Error in ms (mean +/- SD)'),set(gca,'ylim',[-5 80])

subplot(3,1,3),
kjm_errbar_grps(pct_err.',NaN*0*disc_corr.',NaN*0*disc_corr.')
set(gca,'ylim',[0 .4]),box off,set(gca,'ytick',[0:.05:.4]),title('Portion of predictions that were incorrect'),
set(gca,'xtick',1:length(subjects)),set(gca,'xticklabel',subjects),ylabel('Fraction')
set(gca,'ygrid','on')
% legend('ERP - Raw Potential','','','','ERBB - Broadband','','','','Both ERP and ERBB')%,'Location','NorthEastOutside')



%% discrete classification

for k=1:length(subjects)
for m=1:3 % cycle through classes
%%
    load(['data/' subjects{k} '/' subjects{k} '_' cls_all{m} '_pred'],'fh_disc_pred_events','test_events_fold')
    disc_test=[fh_disc_pred_events{1}(:,1); fh_disc_pred_events{2}(:,1); fh_disc_pred_events{3}(:,1)]==[test_events_fold{1}(:,2); test_events_fold{2}(:,2); test_events_fold{3}(:,2)];
    disc_corr(k,m)=sum(disc_test)/length(disc_test);
end
end

%%
figure,kjm_errbar_grps(disc_corr.',NaN*0*disc_corr.',NaN*0*disc_corr.')
box off,set(gca,'ygrid','on')
set(gca,'ylim',[.8 1.00001]),box off,set(gca,'ytick',[.8:.05:1]),title('Portion correct'),ylabel('Fraction')
set(gca,'xtick',1:length(subjects)),set(gca,'xticklabel',subjects),ylabel('Fraction')

