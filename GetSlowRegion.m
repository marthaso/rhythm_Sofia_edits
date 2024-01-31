function [slowPercentage,normalPercent,regionImg,usabelRegionPercentage,...
            countSlowCondtionAreas,...
            statsSlowMean,statsMean,LineLikeMean,FrakturedPatternMean,...
        statsSlow,LineLike,FrakturedPattern] =...
    GetSlowRegion (CVmatrix, lowerlimit)

if ~exist('lowerlimit','var'), lowerlimit=0.2; end

catCV=CVmatrix*0+1;
catCV((CVmatrix(:)<lowerlimit)& (CVmatrix(:)>-0.0))=2;
catCV(CVmatrix(:)>=lowerlimit)=3;
regionImg=catCV;
% if isscalar( slowPercentage) || ...
%         isscalar(normalPercent)  || ...
%         isscalar(usabelRegionPercentage) 
     totalPixelGood=sum(catCV(:) >1); 
% end
% 
% 
% if isscalar( slowPercentage)
     slowPercentage=sum(catCV(:)==2)/totalPixelGood;
% end
% 
% if isscalar(normalPercent)    
     normalPercent=sum(catCV(:)==3)/totalPixelGood;
% end
% 
% if isscalar(usabelRegionPercentage)    
     usabelRegionPercentage=totalPixelGood/size(catCV(:),1);
% end
% 

%Find connected components
bwMatrix=zeros(size(catCV));
bwMatrix(catCV(:)==2)=1;
CC = bwconncomp(bwMatrix);

statsSlow = regionprops('table',CC,'Centroid',...
'MajorAxisLength','MinorAxisLength','Perimeter','EulerNumber','Area');
statsSlow.ratio=statsSlow.MajorAxisLength(:)./statsSlow.MinorAxisLength(:);
statsSlow.gender=abs(statsSlow.EulerNumber-1);
statsSlow=statsSlow(statsSlow.Area>=2,:);
countSlowCondtionAreas=size(statsSlow,1);
% use on larger areas from here on 
areaSize=8;
ratioForLines=4;
genderForFractur=10;
stats=statsSlow(statsSlow.Area>areaSize,:);

LineLike=stats(stats.ratio>ratioForLines,:);
FrakturedPattern=stats(stats.gender>genderForFractur,:);

%Get the averrages
st=@(x) [mean(x),std(x)];
stCon = @(x) table(st(x.Area),st(x.ratio),st(x.gender),'VariableNames', {'Area','ratio','gender'});

statsSlowMean =stCon(statsSlow);
statsMean =stCon(stats);
LineLikeMean =stCon(LineLike);
FrakturedPatternMean =stCon(FrakturedPattern);
%mycell = cell(1,nargout('GetSlowRegion'));
%[mycell{:}] = GetSlowRegion();
end