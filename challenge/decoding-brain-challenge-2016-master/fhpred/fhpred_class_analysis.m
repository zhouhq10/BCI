function data_out=fhpred_class_analysis(subject,cls)


%% load data
    load(['data/' subject '/' subject '_' cls '_pred'])
    
    
%% parameters
    maxhittime=160; %points (ms) within to be considered a 'hit'

%% analysis of continuous data
    %initialize
    temp_hit_fold=[]; %number correct by fold to make sure no dramatic imbalance
    temp_false_fold=[]; % to make sure not a dramatic imbalance by each fold total false ID - house is 1st col, face is 2nd 
    evpred=[]; %predicted events: time (sample) - class - post prob 
    evlink=[]; %linked event to each 'correct' predicted event: time error - class
    evtest=[]; % actual events -- time - class
    evfalse=[]; %false events (see definitions of columns below)
    
for cf=1:3
    %%
    evpred0=fh_cont_pred_events{cf}; % predicted events -- time (sample) - class - post prob 
    evtest0=test_events_fold{cf}; % actual events -- time - class
    evlink0=zeros(size(evpred0,1),2); % linked events

    %% loop through and associate events with closest time
    for k=1:size(evpred0,1)
        %
        tdist=abs(evtest0(:,1)-evpred0(k,1)); [y,ind(k)]=min(tdist); % find the closest actual event (note that if it is flanked by two at equal length, it will take the closest one
        evlink0(k,:)=[[evpred0(k,1)-evtest0(ind(k),1)] evtest0(ind(k),2)]; % timing difference and correct class of the actual timing that this predicted event is associated with
    end
    
    %% hits and misses in each fold - 
    
    % temporal false times -- predicted, but no event there - NOTE:    
        false_inds=or((abs(evlink0(:,1))>maxhittime), (evlink0(:,2)~=evpred0(:,2))); %more than max time away from event OR wrong class
        temp_false_fold=[temp_false_fold; [...
            sum(and(false_inds,evpred0(:,2)==1)) ... %house predicted
            sum(and(false_inds,evpred0(:,2)==2)) ... %face predicted
            ]]; 
        evfalse=[evfalse; [...
            evlink0(false_inds,1) ... % temporal difference to closest stim (must be >100)
            evpred0(false_inds,2) ... % predicted class of false events
            evlink0(false_inds,2) ... % class of closest (linked stim)
            evpred0(false_inds,3) ... % posterior probability of prediction (expect to be high)
            ]]; 
        
    % dispose of false times
        evpred0(false_inds,:)=[];
        ind(false_inds)=[];
        evlink0(false_inds,:)=[];
    
        temp_hit_fold=[temp_hit_fold; [...
            length(intersect(ind, find(evtest0(:,2)==1) ))...  %correct house predicted
            sum(evtest0(:,2)==1)...  %total house stims
            length(intersect(ind, find(evtest0(:,2)==2) ))...  %correct face predicted
            sum(evtest0(:,2)==2)...  %total face stims
            ]];
    
    %store values
        evlink=[evlink; evlink0];
        evpred=[evpred; evpred0];
        evtest=[evtest; evtest0];
    
end

clear y p tdist h k ind *0 ans cf

%%
data_out.evlink=evlink;
data_out.evpred=evpred;
data_out.evtest=evtest;
data_out.evfalse=evfalse;
data_out.subject=subject;
data_out.cls=cls;


%% RESULTS TABLE for results table - continuous classification -- take transpose after moving to excel so rows are total - face - house

disp('-------------------')
disp([subject ' - ' cls])

% number / % of stimuli captured - total - face house
% missed stimuli / % total - face - house
    disp(' ')
    disp(['number of captured stimuli from continuous data stream - '])
    disp([ '     ' ...    
        'all:  ' num2str(sum(sum(temp_hit_fold(:,[1 3])))) ' / ' num2str(size(evtest,1)) ', ' ...
        ]), disp([ '     ' ... 
        'face: ' num2str(sum(and(evlink(:,2)==evpred(:,2),evlink(:,2)==2))) ' / ' num2str(sum(evtest(:,2)==2)) ', '...
        ]), disp([ '     ' ... 
        'house: ' num2str(sum(and(evlink(:,2)==evpred(:,2),evlink(:,2)==1))) ' / ' num2str(sum(evtest(:,2)==1)) ...
        ])

% false predictions (i.e. those > maxhittime from any stim onset) total - face - house
    disp(' ')
    disp(['number of false predictions, by time or class, from continuous data stream - '])
    disp([ '     ' ...    
        'total ' num2str(size(evfalse,1)) '; ' ...
        ]), disp([ '     ' ... 
        'face: time - ' num2str(sum(and(abs(evfalse(:,1))>maxhittime, evfalse(:,2)==2))) ...
        '/' num2str(sum(evfalse(:,2)==2)) ', '...
        'class - ' num2str(sum(and(evfalse(:,2)~=evfalse(:,3), evfalse(:,2)==2))) ... 
        '/' num2str(sum(evfalse(:,2)==2)) '; '...
        ]), disp([ '     ' ... 
        'house: time - :  ' num2str(sum(and(abs(evfalse(:,1))>maxhittime, evfalse(:,2)==1))) ...
        '/' num2str(sum(evfalse(:,2)==1)) ', '...
        'class - ' num2str(sum(and(evfalse(:,2)~=evfalse(:,3), evfalse(:,2)==1))) ... 
        '/' num2str(sum(evfalse(:,2)==1)) ', '...        
        ])
%
 
% temporal accuracy - mean - total - face - house
% temporal accuracy - std - total - face - house
% temporal bias - pvalue
[ha,pa]=ttest(evlink(:,1)); % examine for temporal bias vs zero - All captured
[hf,pf]=ttest(evlink(evlink(:,2)==2,1)); % examine for systematic temporal bias vs zero - Face captured
[hh,ph]=ttest(evlink(evlink(:,2)==2,1)); % examine for systematic temporal bias vs zero - House captured

    disp(' ')
    disp(['average absolute value of temporal error - '])
    disp([ '     ' ...    
        'all:  ' num2str(round(mean(abs(evlink(:,1))))) 'ms +/-' num2str(round(std(abs(evlink(:,1))))) ', ' ...
                '(offset ' num2str(round(mean((evlink(:,1))))) ', Pr(SysErr)=' num2str(round(round((1-pa)*100))) '%); ' ...
        ]), disp([ '     ' ... 
        'face: ' num2str(round(mean(abs(evlink(evlink(:,2)==2,1))))) 'ms +/-' num2str(round(std(abs(evlink(evlink(:,2)==2,1))))) ', ' ...
                '(offset ' num2str(round(mean((evlink(evlink(:,2)==2,1))))) ', Pr(SysErr)=' num2str(round(round((1-pf)*100))) '%); ' ...        
        ]), disp([ '     ' ... 
        'house: ' num2str(round(mean(abs(evlink(evlink(:,2)==1,1))))) 'ms +/-' num2str(round(std(abs(evlink(evlink(:,2)==1,1))))) ', ' ...
                '(offset ' num2str(round(mean((evlink(evlink(:,2)==1,1))))) ', Pr(SysErr)=' num2str(round(round((1-ph)*100))) '%); ' ...
        ])


    
%% Summary for table to export to excel -- semicolon delimited
disp('-------------------')
disp([subject ' - ' cls])



    disp(['number of captured stimuli from continuous data stream - all; face; house; '...
        'number of false predictions, by time or class, from continuous data stream - number total false; face-time; face-class; house-time; house-class'])
    disp([cls '; '...
        '; ' ...% space cell
        num2str(sum(sum(temp_hit_fold(:,[1 3])))) ' / ' num2str(size(evtest,1)) '; ' ...
        num2str(sum(and(evlink(:,2)==evpred(:,2),evlink(:,2)==2))) ' / ' num2str(sum(evtest(:,2)==2)) '; '...
        num2str(sum(and(evlink(:,2)==evpred(:,2),evlink(:,2)==1))) ' / ' num2str(sum(evtest(:,2)==1)) '; ' ... %end  of -- portion stimuli captured - total - face house
        '; ' ...% space cell
        num2str(size(evfalse,1)) '; ' ...
        num2str(sum(and(abs(evfalse(:,1))>maxhittime, evfalse(:,2)==2))) '/' num2str(sum(evfalse(:,2)==2)) '; '...
        num2str(sum(and(evfalse(:,2)~=evfalse(:,3), evfalse(:,2)==2))) '/' num2str(sum(evfalse(:,2)==2)) '; '...
        num2str(sum(and(abs(evfalse(:,1))>maxhittime, evfalse(:,2)==1))) '/' num2str(sum(evfalse(:,2)==1)) '; '...
        num2str(sum(and(evfalse(:,2)~=evfalse(:,3), evfalse(:,2)==1))) '/' num2str(sum(evfalse(:,2)==1)) '; '... % end of -- false predictions (i.e. those > 100ms from any stim onset) total - face - house  
        '; '... % space cell
        num2str(round(mean(abs(evlink(:,1))))) 'ms +/-' num2str(round(std(abs(evlink(:,1))))) ', ' ...
                '(offset ' num2str(round(mean((evlink(:,1))))) '); ' ...
        num2str(round(mean(abs(evlink(evlink(:,2)==2,1))))) 'ms +/-' num2str(round(std(abs(evlink(evlink(:,2)==2,1))))) ', ' ...
                '(offset ' num2str(round(mean((evlink(evlink(:,2)==2,1))))) '); ' ...        
        num2str(round(mean(abs(evlink(evlink(:,2)==1,1))))) 'ms +/-' num2str(round(std(abs(evlink(evlink(:,2)==1,1))))) ', ' ...
                '(offset ' num2str(round(mean((evlink(evlink(:,2)==1,1))))) '); ' ... % end of 'average absolute value of temporal error - '
        ])


