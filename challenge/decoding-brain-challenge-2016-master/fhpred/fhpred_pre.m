function fhpred_pre(subject,cls)
% function fhpred_pre(subject,cls) 
% this function does pre-processing prior to classifier steps

%% load data
switch cls
    
    case 'erp'
        %  load raw data, reject bad chans, car
        load(['data/' subject '/' subject '_faceshouses'])  
        signal=car(data); clear data
        
    case 'bb'
        load(['data/' subject '/' subject '_fh_bb'])  
        signal=bb; clear bb
end

%% create events vector    
    [events]=fh_get_events(stim);
    events(find(events(:,3)==0),:)=[];
    events(:,2)=[];


%% defaults
    srate=1000;
    tlength=size(signal,1);
    num_chans=size(signal,2);
    tlims=[-199 400]; % times to start and end erps
    erp_baseline=[-199 50]; % times to calcualte erp based upon (must be within tlims)
    
%% identify thirds

    % first third
    ind1=1:floor(tlength/3);
    ev1=and(...
        events(:,1)<ind1(end),...
        events(:,1)>ind1(1));
    ev1=events(ev1,:);

    % second third
    ind2=(floor(tlength/3)+1):floor(tlength*2/3);
    ev2=and(...
        events(:,1)<ind2(end),...
        events(:,1)>ind2(1));
    ev2=events(ev2,:);

    % third third
    ind3=(floor(tlength*2/3)+1):tlength;
    ev3=and(...
        events(:,1)<ind3(end),...
        events(:,1)>ind3(1));
    ev3=events(ev3,:);

%% 3 times cross-folding - divide into train and test segments
for cf=1:3
    disp(['Fold #' num2str(cf) ' of 3']) 
    switch cf
        case 1
            % get test event indices
            test_t =ind3;
            train_t=[ind1 ind2];
            test_e =[ev3(:,1)-ind2(end) ev3(:,2)];
            train_e=[ev1; ev2];
        case 2
            % get test event indices
            test_t =ind2;
            train_t=[ind1 ind3];
            test_e =[ev2(:,1)-ind2(1) ev2(:,2)];
            train_e=[ev1;...
                [ev3(:,1)-ind2(end)+ind1(end) ev3(:,2)]...
            ];
        case 3
            % get test event indices
            test_t =ind1;
            train_t=[ind2 ind3];
            test_e =[ev1];
            train_e=[...
                [ev2(:,1)-ind1(end) ev2(:,2)];...
                [ev3(:,1)-ind2(end)+length(ind2) ev3(:,2)]...
            ];            
    end

%% scale by std, etc (based only on train segments) 
    for k=1:num_chans
        signal(:,k)=signal(:,k)/std(signal(train_t,k)); % signal(:,k)=(signal(:,k)-mean(signal(train_t,k)))/std(signal(train_t,k));
    end

%% get class specific {face, house} STA templates from train - zero out for 100 ms pre-stim
   
    % get sta templates
    sta_h=fh_sta(signal(train_t,:), train_e, 1, tlims); % houses
    sta_f=fh_sta(signal(train_t,:), train_e, 2, tlims); % faces
    
    % recenter stas w.r.t. baseline
    for k=1:num_chans
        sta_h(:,k)=sta_h(:,k)-mean(sta_h((erp_baseline(1):erp_baseline(2))-tlims(1)+1,k));
        sta_f(:,k)=sta_f(:,k)-mean(sta_f((erp_baseline(1):erp_baseline(2))-tlims(1)+1,k));
    end

%% generate train data - 
%  get samples of template at appropriate point and at every 100ms for each class - 
%  note that this method is not efficient, because templates are class specific. 

% NOTE: had to make this deterministic for microsoft competition redux
% error('a','a')
    %  generate training points between each stimulus, 
    %  must be min 100ms from any actual stimulus point and min 50 from each other
    a=train_e(:,1);
    a=sort(a,'ascend');
    npts=[];
    for k=2:length(a)
        b=randperm(floor((a(k)-a(k-1)-200)/100)); % floor(rand(4,1)*(a(k)-a(k-1)-200))
        if length(b)>3
        b=100*b(1:4).'+floor(50*rand(4,1))+a(k-1);
        else
            b=100*b(1:length(b)).'+floor(50*rand(length(b),1))+a(k-1);
        end        
        npts=[npts; b];
    end
    
    train_e=[train_e; [npts 0*npts]]; clear  a b npts
    
    % get projection into templates
    f_template_train=zeros(size(train_e,1),num_chans);
    h_template_train=0*f_template_train;
    dt0=signal(train_t,:); % select training data
    for k=1:size(train_e,1) % dot products
        for chan=1:num_chans            
            dt=dt0(train_e(k,1)+[tlims(1):tlims(2)],chan); % select data
            dt=dt-mean(dt((erp_baseline(1):erp_baseline(2))-tlims(1)+1)); %baseline data
            f_template_train(k,chan)=sum(sta_f(:,chan).*dt); % convolve
            h_template_train(k,chan)=sum(sta_h(:,chan).*dt); % convolve
        end
    end

%% generate test data - note that loop method sucks in matlab
    f_template_test=zeros(length(test_t)-(tlims(2)-tlims(1)),num_chans);
    h_template_test=0*f_template_test;

    for k=1:(-tlims(1)) % first few points - zero pad
        for chan=1:num_chans
%             dt=[zeros(-(tlims(1)+k)+1,1); signal(1:(tlims(2)+k),chan)]; % select data
            dt=[zeros(-(tlims(1)+k)+1,1); signal(test_t(k):(test_t(k)+tlims(2)+k-1),chan)];  % select data
            dt=dt-mean(dt((erp_baseline(1):erp_baseline(2))-tlims(1)+1)); %baseline data
            f_template_test(k,chan)=sum(sta_f(:,chan).*dt); % convolve
            h_template_test(k,chan)=sum(sta_h(:,chan).*dt); % convolve
        end
    end    
    %
    for k=(1-tlims(1)):size(f_template_test,1) % note that these times line up with a phase lag according to tlims(1)
        for chan=1:num_chans
%             dt=signal(k+[tlims(1):tlims(2)],chan); % select data
            dt=signal(test_t(k)+[tlims(1):tlims(2)],chan); % select data
            dt=dt-mean(dt((erp_baseline(1):erp_baseline(2))-tlims(1)+1)); %baseline data
            f_template_test(k,chan)=sum(sta_f(:,chan).*dt); % convolve
            h_template_test(k,chan)=sum(sta_h(:,chan).*dt); % convolve
        end
    end

%% store in appropriate fold, 
    f_template_test_fold{cf} =f_template_test;
    h_template_test_fold{cf} =h_template_test;
    %
    f_template_train_fold{cf}=f_template_train;
    h_template_train_fold{cf}=h_template_train;
    %
    train_events_fold{cf}=train_e;
    test_events_fold{cf}=test_e;
    %
    sta_f_fold{cf}=sta_f;
    sta_h_fold{cf}=sta_h;
    %
    stim_train_fold{cf}=stim(train_t);
    stim_test_fold{cf}=stim(test_t(1:size(f_template_test,1)));

end

%% save testing and training data for later classification 
save(['data/' subject '/' subject '_' cls '_cross_folds'],'*fold*')
