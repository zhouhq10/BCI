function fhpred_datashuffle(subject)

%% This function shuffles the data from each experiment run to obtain a datastream with random sequence and random isi's between 1-800ms

load(['data/' subject '/' subject '_faceshouses.mat'])

if subject=='ha', stim(85480:85482)=0; stim(165880:165882)=0; end

%%

tmp=hann(100);
en_win=tmp(51:100); en_win(en_win==0)=0.0002;
st_win=tmp(1:50); st_win(st_win==0)=0.0002;

%% procedure

% A random excision point is picked during the inter-stimulus interval following each stimulus,
% and then the sequence of stimuli is shuffled temporally, 
% At these excision points, data are tapered on either end of this excision.


%%
    evs=fh_get_events(stim);
    evs(evs(:,3)>0,:)=[];
    isi_st=evs(:,1);
    
%% inter-trial intervals
    % get data epochs
    iti0_data=data(1:isi_st(1),:);
    iti1_data=data((isi_st(101)+1):isi_st(102),:);
    iti2_data=data((isi_st(202)+1):isi_st(203),:);
    iti3_data=data((isi_st(303)+1):end,:);
    
    % add window-tapers to data epochs
    iti0_data([-49:0]+end,:)=iti0_data([-49:0]+end,:).*repmat(en_win,1,size(data,2)); 
    %
    iti1_data(1:50,:)=iti1_data(1:50,:).*repmat(st_win,1,size(data,2));
    iti1_data([-49:0]+end,:)=iti1_data([-49:0]+end,:).*repmat(en_win,1,size(data,2));
    %
    iti2_data(1:50,:)=iti2_data(1:50,:).*repmat(st_win,1,size(data,2));
    iti2_data([-49:0]+end,:)=iti2_data([-49:0]+end,:).*repmat(en_win,1,size(data,2));    
    %
    iti3_data(1:50,:)=iti3_data(1:50,:).*repmat(st_win,1,size(data,2));
    iti3_data([-49:0]+end,:)=iti3_data([-49:0]+end,:).*repmat(en_win,1,size(data,2));    
    
    % get stim epochs
    iti0_stim=stim(1:isi_st(1));
    iti1_stim=stim((isi_st(101)+1):isi_st(102));
    iti2_stim=stim((isi_st(202)+1):isi_st(203));
    iti3_stim=stim((isi_st(303)+1):end,:);
    
%% do exicisions    
    clip_pts=floor(400*[0; rand(99,1); 0; 0; rand(99,1); 0; 0; rand(99,1); 0])+isi_st;
    
    for k=1:100
        % isolate data
        dtmp=data(((clip_pts(k)+1):clip_pts(k+1)),:);
        % add window-tapers to data epoch
        dtmp(1:50,:)=dtmp(1:50,:).*repmat(st_win,1,size(data,2));
        dtmp([-49:0]+end,:)=dtmp([-49:0]+end,:).*repmat(en_win,1,size(data,2));   
        %
        data_excisions{k}=dtmp;
        stim_excisions{k}=stim(((clip_pts(k)+1):clip_pts(k+1)));
    end
    
    for k=102:201
        % isolate data
        dtmp=data(((clip_pts(k)+1):clip_pts(k+1)),:);
        % add window-tapers to data epoch
        dtmp(1:50,:)=dtmp(1:50,:).*repmat(st_win,1,size(data,2));
        dtmp([-49:0]+end,:)=dtmp([-49:0]+end,:).*repmat(en_win,1,size(data,2));   
        %
        data_excisions{k-1}=dtmp;
        stim_excisions{k-1}=stim(((clip_pts(k)+1):clip_pts(k+1)));
    end

    for k=203:302
        % isolate data
        dtmp=data(((clip_pts(k)+1):clip_pts(k+1)),:);
        % add window-tapers to data epoch
        dtmp(1:50,:)=dtmp(1:50,:).*repmat(st_win,1,size(data,2));
        dtmp([-49:0]+end,:)=dtmp([-49:0]+end,:).*repmat(en_win,1,size(data,2));   
        %
        data_excisions{k-2}=dtmp;
        stim_excisions{k-2}=stim(((clip_pts(k)+1):clip_pts(k+1)));
    end

    
    %% seam together shuffled data
    rand_seq=[randperm(100) 100+randperm(100) 200+randperm(100)];    
    
    data_shuffle=[]; stim_shuffle=[]; 
    
    % concatenate shuffled data
    data_shuffle=[data_shuffle; iti0_data];
    for k=1:100
    data_shuffle=[data_shuffle; data_excisions{rand_seq(k)}];
    end
    data_shuffle=[data_shuffle; iti1_data];
    for k=101:200
    data_shuffle=[data_shuffle; data_excisions{rand_seq(k)}];
    end    
    data_shuffle=[data_shuffle; iti2_data];
    for k=201:300
    data_shuffle=[data_shuffle; data_excisions{rand_seq(k)}];
    end    
    data_shuffle=[data_shuffle; iti3_data];
    % concatenate stim
    stim_shuffle=[stim_shuffle; iti0_stim];
    for k=1:100
    stim_shuffle=[stim_shuffle; stim_excisions{rand_seq(k)}];
    end
    stim_shuffle=[stim_shuffle; iti1_stim];
    for k=101:200
    stim_shuffle=[stim_shuffle; stim_excisions{rand_seq(k)}];
    end    
    stim_shuffle=[stim_shuffle; iti2_stim];
    for k=201:300
    stim_shuffle=[stim_shuffle; stim_excisions{rand_seq(k)}];
    end    
    stim_shuffle=[stim_shuffle; iti3_stim];    
%%    


data=data_shuffle; stim=stim_shuffle;
save(['data/' subject '/' subject '_faceshouses_shuffled.mat'],'stim','data','srate')


