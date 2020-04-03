function pc1=kjm_lnA_timecourse(data,pcvec1,srate,freqs)
% function tf=kjm_tf_pwr(data,srate,freqs) generate complex time-frequency
%   analysis in range freqs
%

%%

tfp=zeros(length(data),length(freqs)); %time-frequency power
    
for tmp=1:length(freqs)
    freq=freqs(tmp);

    %% create morlet wavelet
    t=1:floor(5*srate/freq); tmid=floor(max(t)/2);
    wvlt=exp(1i*2*pi*(freq/srate)*(t-tmid)).*exp(-((t-tmid).^2)/(2*(srate/freq)^2)); %gaussian envelope
%     figure, plot(real(wvlt)), hold on, plot(imag(wvlt),'r'), hold on, plot(abs(wvlt),'g') %if you want to have a look at wavelet

    %% calculate convolution
    tconv=conv(wvlt,data); % convolution
    tconv([1:(floor(length(wvlt)/2)-1) floor(length(tconv)-length(wvlt)/2+1):length(tconv)])=[]; %eliminate edges 
    tconv=abs(tconv).^2; % power
    if mean(tconv)==0, error('mean power 0','mean power 0'),end  %if there is some problem
    tconv=tconv/mean(tconv);  %normalize power at this freq 
    tfp(:,tmp)=tconv;    
    
end

pc1=log(tfp)*pcvec1;