function fhpred_cls_all=fhpred_master(subject)


% function fhpred_master(subject)
%     addpath /toolbox/ctmr/
%     addpath ../../kai_toolbox
    addpath dc_files
    addpath xs_files
    
    warning('off','signal:psd:PSDisObsolete'); %annoying

%% area labels -- See Destrieux et al, "Automatic parcellation of human cortical gyri and sulci using standard anatomical nomenclature", NeuroImage, 2010, for how areas were labeled (labeling done by hand based upon MRI cross-section)
% these labels are in the variable elcode of the #subject#_xslocs.mat of
% "locs" folder, corresponding to electrode/channel number 
area_lbls={...
	'Temporal pole', ... %1
	'Parahippocampal gyrus',... %2, parahippocampal part of the medial occipito-temporal gyrus
	'Inferior temporal gyrus',... %3
	'Middle temporal gyrus',... %4
	'fusiform gyrus',... %5 Lateral occipito-temporal gyrus, 
	'Lingual gyrus',... %6, lingual part of  the medial occipito-temporal gyrus
	'Inferior occipital gyrus',... %7
	'Cuneus',... %8
	'Post-ventral cingulate gyrus',... %9 Posterior-ventral part of the 
	'Middle Occipital gyrus',... %10 
	'occipital pole',... %11 
	'precuneus',... %12 
	'Superior occipital gyrus',... %13 
	'Post-dorsal cingulate gyrus',... %14 Posterior-dorsal part of the cingulate gyrus
    ' ',...%15
    ' ',...%16    
    ' ',...%17
    ' ',...%18
    ' ',...%19
    'Non-included area',... %20
    };

    % NOTE - the electrodes for plotting and the plotting range for the MRI
    % are in #subject#_xslocs.mat of "locs" folder. The corresponding MRIs are 
    % in the "brains" folder. The functions to plot these locations are in
    % the xs_files folder (use the xs_disp.m function to start)

%% all class types
    cls_all= {'erp','bb','bth'};

%% load data
    disp(['Subject: ' subject])
    load(['data/' subject '/' subject '_faceshouses']) %note that bad channels have already been rejected

%% re-reference / regress out 1st mode
    data=car(data); disp('common average reference ... ') % common average reference
%     data=pcr(data); % regress out first mode

%% get events 
    pts=fh_get_events(stim);
    
%% decouple spectral motifs
    disp('decoupling ...')
    [spectra]=calc_dg_spectra(data,pts); %spectral snapshots
    [nspectra]=calc_nspectra(spectra); %normalize spectra

    
    [pc_weights, pc_vecs, pc_vals, f]=dg_pca_step(nspectra); %perform PCA
%     save(['data/' subject '/' subject '_fh_decoupled'], 'pc_*', '*spectra','f','pts') %save if needed, to examine spectra, etc
    clear('pc_weights','pc_vals', '*spectra','f') % cleanup, leaving pc_vecs    
    disp('...')

%% get broadband timecourse from 1st spectral principle component
    disp('Getting broadband timecourse ...')
    
    % create indices to exclude around harmonics of 60Hz
    f0=1:200; no60=[];
    for k=1:ceil(max(f0/60)), no60=[no60 (60*k-3):(60*k+3)]; end %3 hz up or down 
    f0=setdiff(f0,no60); %dispose of 60hz stuff

    % get lnA
    lnA=0*data;
    for chan=1:size(data,2)
        disp([subject ' channel ' num2str(chan) ' / ' num2str(size(data,2))])
        dt=data(:,chan); 
        mm=squeeze(pc_vecs(:,chan,:))';  %mixing matrix
        pcvec1=mm(:,1);  
        lnA(:,chan)=kjm_lnA_timecourse(dt,pcvec1,srate,f0); % this generates lnA timeseries for each channel
    end
    
    % smooth lnA, exponentiate, and subtract 1 to get bb timecourse
    bb=fh_pc_clean(lnA); bb=bb-1; clear lnA, % smooth, zscore, re-exponentiate, subtract 1
    save(['data/' subject '/' subject '_fh_bb'],'bb','stim'), 


%% generate convolutions / pre-process data
% 
for m=1:2 % cycle through template types
%     disp(cls_all{m})
%     
% %     % pre-process - calculate cross folds
    fhpred_pre(subject,cls_all{m}) % generate convolved data, with 3-fold cross-folding    

end

%% perform classifications (discrete & continuous)
for m=1:3 % cycle through classes
%     disp(cls_all{m})
    fhpred_classification(subject,cls_all{m});
    fhpred_cls_all{m}=fhpred_class_analysis(subject,cls_all{m});    

end