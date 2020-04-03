function [evs]=fh_get_events(stim)
%this function defines evs lying at the beginning (1st column)and middle
%(2nd column) of each stimulus and isi period
% 0 = ISI
% 1 = HOUSE STIM
% 2 = FACE STIM

b=find((stim-[0; stim(1:(length(stim)-1))])~=0);
c=floor(diff(b)/2);
b(end)=[];
d=b+c;
evs(:,1)=b;
evs(:,2)=d;
evs(:,3)=stim(d);
evs(find(evs(:,3)==0),:)=[];
evs(find(evs(:,3)<51),3)=1;
evs(find(evs(:,3)==101),3)=0;
evs(find(evs(:,3)>50),3)=2;
clear b c d

%clip if too close to ends of run files
evs(find(or(evs(:,1)<500,evs(:,2)>(length(stim)-1000))),:)=[];
