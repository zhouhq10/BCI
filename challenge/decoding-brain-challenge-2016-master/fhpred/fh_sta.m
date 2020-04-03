function sta=fh_sta(data, events, fh_class, tlims)
% function sta=fh_sta(data, events, fh_class)
% this function outputs stas for each channel
% input variables:
%     data(time,channel) - the recorded channel data
%     events(event,3) - 1st column - onset times; 2nd column - stimulus type
%     fh_class - 1 - Faces; 2 - Houses; 0 - ISI (from 3rd column of events)
%     tlims(1,2) - 1st element start time (+ or -) relative to event 
%                - 2nd element end time relative to event
% 
% output variable
%     - sta(tlims(1):tlims(2),channel) - output sta    
% 
% kjm 7/2010    


cls_times=events(find(events(:,2)==fh_class),1);

sta=zeros((tlims(2)-tlims(1)+1),size(data,2));

for k=1:length(cls_times)
    sta=sta+data(cls_times(k)+[tlims(1):tlims(2)],:);
end

sta=sta/k;
    
    
    
    