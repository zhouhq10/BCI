function fhpred_classification(subject,cls)




%% classification parameters

    % for both discrete and continuous classification
    clsparms.preselect_r2=.05;
    clsparms.disc_type='linear';

    % for continuous classification
    clsparms.minprob=.51; % minimum post probability to be included as event
    clsparms.minTdist=320; %minimum distance in time, units samplesize
    clsparms.sm_pp='y'; % smooth posterior probability prior to finding peaks



%% load data - this cell is where need to downselect channels by anatomy, etc, if that is done

    if strcmp(cls,'bth') %load both
        
        % load bb, store
        load(['data/' subject '/' subject '_bb_cross_folds'],'*fold*')
        f_template_train_fold_bb=f_template_train_fold; %train data for face templates
        h_template_train_fold_bb=h_template_train_fold; %train data for house templates
        f_template_test_fold_bb=f_template_test_fold; %test data for face templates
        h_template_test_fold_bb=h_template_test_fold; %test data for house templates
        
        % load erp
        load(['data/' subject '/' subject '_erp_cross_folds'],'*fold*')
        
        % stitch together for each fold
        for cf=1:3
            f_template_train_fold{cf}=[f_template_train_fold_bb{cf} f_template_train_fold{cf}]; %train data for face templates
            h_template_train_fold{cf}=[h_template_train_fold_bb{cf} h_template_train_fold{cf}]; %train data for house templates
            f_template_test_fold{cf}=[f_template_test_fold_bb{cf} f_template_test_fold{cf}]; %test data for face templates
            h_template_test_fold{cf}=[h_template_test_fold_bb{cf} h_template_test_fold{cf}]; %test data for house templates
        end
        clear *_bb
        
    else %load bb or erp        
        load(['data/' subject '/' subject '_' cls '_cross_folds'],'*fold*')
        
    end
    % load stim
    load(['data/' subject '/' subject '_faceshouses'],'stim')
    
%% get timing estimates with posterior probs and logp for continuous, and predict for supervised timing
for cf=1:3
    % call discrete classifier looped 
    [fh_disc_pred_events{cf}]=...
    fhpred_discrete_folds(...
        f_template_train_fold{cf}, ... %train data for face templates
        h_template_train_fold{cf}, ... %train data for house templates
        train_events_fold{cf}(:,2), ... %labels of training data
        f_template_test_fold{cf}(test_events_fold{cf}(:,1),:), ... %test data for face templates
        h_template_test_fold{cf}(test_events_fold{cf}(:,1),:), ... %test data for house templates
        test_events_fold{cf}(:,2), ... %labels of testing data
        clsparms ...
        );
    title([subject ' r2 values by electrode, with cutoff, class=' cls ', fold #' num2str(cf)])
    
    % call continuous classifier looped 
    [fh_cont_pred_events{cf}]=...
    fhpred_continuous_folds(...
        f_template_train_fold{cf}, ... %train data for face templates
        h_template_train_fold{cf}, ... %train data for house templates
        train_events_fold{cf}, ... %labels of training data
        f_template_test_fold{cf}, ... %test data for face templates
        h_template_test_fold{cf}, ... %test data for house templates
        test_events_fold{cf}, ... %labels of testing data
        stim_test_fold{cf}, ... %stimulus, to cut times in between runs (when talking with researchers, etc)
        clsparms ...
        );
    title([subject ' r2 values by electrode, with cutoff, class=' cls ', fold #' num2str(cf)])
end


%% save
save(['data/' subject '/' subject '_' cls '_pred'],'*_cont_*','*_disc_*', 'test_events*')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fh_disc_pred_events]=fhpred_discrete_folds(traindata_f,traindata_h,trainlabels,testdata_f,testdata_h,testlabels,clsparms)

%% parameters to use    
    preselect_r2=clsparms.preselect_r2; %r2 threshold to keep channels
    disc_type=clsparms.disc_type; %classifier discriminant analysis type


        %% down-select to discriminable channels only for feature space - decided not to explicitly include face vs house
        for k=1:size(traindata_f,2)
            rf0(k)=rsa(traindata_f(find(trainlabels==2),k),traindata_f(find(trainlabels==0),k)); % 2 is face
            rh0(k)=rsa(traindata_h(find(trainlabels==1),k),traindata_h(find(trainlabels==0),k)); % 1 is house
        end
        f2u=find(abs(rf0)>preselect_r2);
        h2u=find(abs(rh0)>preselect_r2);
        
        %plot
        figure, plot(rf0,'b*')
        hold on, plot(rh0,'rs')
        hold on, plot([0 size(traindata_f,2)],preselect_r2*[1 1],'k-')
        xlabel('electrode number (concat if both BB and ERP)'), ylabel('signed r2')
        
        legend('face feat','house feat', 'cutoff','Location','NorthEastOutside')
        
        %% can change this later to be customized classifier if desired
    
    [fh_class_out, err_out,fh_post_prob,fh_logp] = classify(...
        [testdata_f(:,f2u) testdata_h(:,h2u)],...
        [traindata_f(find(trainlabels>0),f2u) traindata_h(find(trainlabels>0),h2u)],...
        trainlabels(find(trainlabels>0)),disc_type); 
    
    tmp=0*fh_class_out;
    
    for k = 1:size(fh_class_out)
        tmp(k)=fh_post_prob(k,fh_class_out(k));
    end
    fh_disc_pred_events=[fh_class_out tmp];
    
    disp(['sup accuracy by fold = ' num2str(round(100*(sum([fh_class_out==testlabels])/length(testlabels)))) '%'])
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fh_cont_pred_events]=fhpred_continuous_folds(traindata_f,traindata_h,trainlabels,testdata_f,testdata_h,testlabels,stim_test,clsparms)

%% parameters to use
    preselect_r2=clsparms.preselect_r2;
    disc_type=clsparms.disc_type; %classifier discriminant analysis type
    minprob=clsparms.minprob; % minimum post probability to be included as event
    minTdist=clsparms.minTdist; %minimum distance in time, units samplesize
    sm_pp=clsparms.sm_pp; % smooth posterior probability prior to finding peaks


    %% down-select to discriminable channels only for feature space - decided not to explicitly include face vs house
        for k=1:size(traindata_f,2)
            rf0(k)=rsa(traindata_f(find(trainlabels(:,2)==2),k),traindata_f(find(trainlabels(:,2)==0),k)); % 2 is face
            rh0(k)=rsa(traindata_h(find(trainlabels(:,2)==1),k),traindata_h(find(trainlabels(:,2)==0),k)); % 1 is house
        end
        f2u=find(abs(rf0)>preselect_r2);
        h2u=find(abs(rh0)>preselect_r2);

        testdata=[testdata_f(:,f2u) testdata_h(:,h2u)];
        traindata=[traindata_f(:,f2u) traindata_h(:,h2u)];
        clear *a_f *a_h        
    %% will change this later to be my own classifier based upon mahalanobis distance

        [fh_class_out, err_out,fh_post_prob,fh_logp] = classify(testdata,traindata,trainlabels(:,2),disc_type);
        for k = 2:3
            fh_post_prob(stim_test==0,k)=0;
        end
    
    
    %% get peaks of posterior probability trace
    if sm_pp=='n' %use smoothed post prob?
    [hpks,htimes] = findpeaks(fh_post_prob(:,2),'MinPeakHeight',minprob,'MinPeakDistance',minTdist); % peaks in post prob by time of house stims
    [fpks,ftimes] = findpeaks(fh_post_prob(:,3),'MinPeakHeight',minprob,'MinPeakDistance',minTdist); % peaks in post prob by time of face stims
    %
    [bpks,btimes] = findpeaks(max(fh_post_prob(:,2:3),[],2),'MinPeakHeight',minprob,'MinPeakDistance',minTdist); % peaks in post prob by time of face stims
    %
    [c,iH,ib]=intersect(htimes,btimes);hpks=hpks(iH); htimes=htimes(iH);
    [c,iF,ib]=intersect(ftimes,btimes);fpks=fpks(iF); ftimes=ftimes(iF);
    clear c i*
    
    figure, plot(fh_post_prob(:,2),'color',[.5 0 0])
    hold on,plot(fh_post_prob(:,3),'color',[0 0 .5])
    hold on, plot(testlabels(testlabels(:,2)==1,1),fh_post_prob(testlabels(testlabels(:,2)==1,1),2),'ro')
    hold on, plot(testlabels(testlabels(:,2)==2,1),fh_post_prob(testlabels(testlabels(:,2)==2,1),3),'bo')
    hold on, plot(htimes,hpks,'r+')
    hold on, plot(ftimes,fpks,'b+')
    
    
    elseif sm_pp=='y'
    
    for k=1:3 %do smoothing with gaussian of same width
        fh_pp_sm(:,k)=log(fh_pc_clean(fh_post_prob(:,k))); 
    end
    
    %
    [hpks,htimes] = findpeaks(fh_pp_sm(:,2),'MinPeakHeight',minprob,'MinPeakDistance',minTdist); % peaks in post prob by time of house stims
    [fpks,ftimes] = findpeaks(fh_pp_sm(:,3),'MinPeakHeight',minprob,'MinPeakDistance',minTdist); % peaks in post prob by time of face stims
    %
    [bpks,btimes] = findpeaks(max(fh_pp_sm(:,2:3),[],2),'MinPeakHeight',minprob,'MinPeakDistance',minTdist); % peaks in post prob by time of face stims
    %
    [c,iH,ib]=intersect(hpks,bpks);hpks=hpks(iH); htimes=htimes(iH);
    [c,iF,ib]=intersect(fpks,bpks);fpks=fpks(iF); ftimes=ftimes(iF);
    
    figure, plot(fh_pp_sm(:,2),'color',[.5 0 0])
    hold on,plot(fh_pp_sm(:,3),'color',[0 0 .5])
    hold on, plot(testlabels(testlabels(:,2)==1,1),fh_pp_sm(testlabels(testlabels(:,2)==1,1),2),'ro')
    hold on, plot(testlabels(testlabels(:,2)==2,1),fh_pp_sm(testlabels(testlabels(:,2)==2,1),3),'bo')
    hold on, plot(htimes,hpks,'r+')
    hold on, plot(ftimes,fpks,'b+')
    end
    
    clear bpks btimes c i* err_out k rf0 rh0 *_cla* *_logp*
    %
    if size(hpks,2)>size(hpks,1)
        fh_cont_pred_events=[[htimes.' 1+0*htimes.' hpks.']; [ftimes.' 2+0*ftimes.' fpks.']]; % predicted events - time - label - post prob
    else
        fh_cont_pred_events=[[htimes 1+0*htimes hpks]; [ftimes 2+0*ftimes fpks]]; % predicted events - time - label - post prob
    end
    
    
    [dsort,isort]=sort(fh_cont_pred_events(:,1),'ascend');
    fh_cont_pred_events=fh_cont_pred_events(isort,:); clear dsort isort
    
    %%

    
 
    