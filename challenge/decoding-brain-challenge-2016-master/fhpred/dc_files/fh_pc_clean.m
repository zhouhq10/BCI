function bb=fh_pc_clean(lnA)
%this function smooths the pcs,


%% %%DO SMOOTHING%%%%%%%
winlength=80;

bb=0*lnA;
for k = 1:size(lnA,2)
    % convolve with gaussian window
    lnAs=(conv(gausswin(winlength),lnA(:,k)));
    % clip edges
    lnAs(1:floor(winlength/2-1))=[];
    lnAs((length(lnAs)-floor(winlength/2-1)):length(lnAs))=[]; 
    %z-score
    lnAs=(lnAs-mean(lnAs))/std(lnAs);    
    %  re-exponentiate (as originally in log)
    bb(:,k)=exp(lnAs); 
end
