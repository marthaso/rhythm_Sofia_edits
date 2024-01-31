function [mask] = GeneratedMask(data,thVal)
%GENERATEDMASK Summary of this function goes here
%   Detailed explanation goes here

%temp2 = diff(data,1,3); % first derivative
temp2=data;
%m=mean(temp2,3);
%mx=max(temp2,[],3);
%mxn=mx-m;

%mx2=max(mxn(:));
%minx=min(mxn(:));
stdX=std(temp2,0,3);
mask=zeros(size(temp2,1),size(temp2,2));
mask(stdX> max(stdX(:))*thVal)=1;
%mask(mx> m +stdX*5*thVal)=1;
%mask(mxn> minx+(mx2-minx)*thVal)=1;
end

