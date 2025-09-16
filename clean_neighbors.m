function [cleanaMap] = clean_neighbors(aMap1,bg,mask3)
% Check all neighbors. If at least 4 of them are more than 10 msec different. Set
% pixel to NA.

for i = 2:255 %255
    for j = 2:255 %255
        our_pixel = aMap1(i,j);
        neighbor1 = aMap1(i-1, j-1);
        neighbor2 = aMap1(i, j-1);
        neighbor3 = aMap1(i+1, j-1);
        neighbor4 = aMap1(i-1, j);
        neighbor5 = aMap1(i+1, j);
        neighbor6 = aMap1(i-1, j+1);
        neighbor7 = aMap1(i, j+1);
        neighbor8 = aMap1(i+1, j+1);
        dif1 = abs(our_pixel - neighbor1);
        dif2 = abs(our_pixel - neighbor2);
        dif3 = abs(our_pixel - neighbor3);
        dif4 = abs(our_pixel - neighbor4);
        dif5 = abs(our_pixel - neighbor5);
        dif6 = abs(our_pixel - neighbor6);
        dif7 = abs(our_pixel - neighbor7);
        dif8 = abs(our_pixel - neighbor8);
        difs = [dif1, dif2, dif3, dif4, dif5, dif6, dif7, dif8];
        counter = 0;
        for p = 1:numel(difs)
            if difs(p) > 5
                counter = counter + 1;
            end
        end
        if counter >= 2
            aMap1(i,j) = NaN;

        end
    end
end


% figure;
% G = real2rgb(bg, 'gray');
% imagesc(G)
% hold on
% contourf(aMap1,max(max(aMap1)),'LineColor','none');
% colormap (flipud(jet));
% c=colorbar;
% axis off
% title('Activation Map')
% c.Label.String = 'Activation Time (ms)';


% Check all neighbors. If at least 4 of them are NaN, turn to NaN. Will
% have to think what to do about edges.
numofneighbors = 2;

for i = numofneighbors+1:256-numofneighbors%256-numofneighbors
    for j = numofneighbors+1:256-numofneighbors%256-numofneighbors
        if isnan(aMap1(i,j))
            neighbors =[];
            for t = 1:numofneighbors
                neighbor1 = aMap1(i-t, j-t);
                neighbor2 = aMap1(i, j-t);
                neighbor3 = aMap1(i+t, j-t);
                neighbor4 = aMap1(i-t, j);
                neighbor5 = aMap1(i+t, j);
                neighbor6 = aMap1(i-t, j+t);
                neighbor7 = aMap1(i, j+t);
                neighbor8 = aMap1(i+t, j+t);
                neighbors = [neighbors, neighbor1, neighbor2, neighbor3, neighbor4, neighbor5, neighbor6, neighbor7, neighbor8];
            end
            neighbor9 = aMap1(i-1, j-2);
            neighbor10 = aMap1(i+1, j-2);
            neighbor11 = aMap1(i-2, j-1);
            neighbor12 = aMap1(i-2, j+1);
            neighbor13 = aMap1(i+2, j+1);
            neighbor14 = aMap1(i-1, j+2);
            neighbors = [neighbors, neighbor9, neighbor10, neighbor11, neighbor12, neighbor13, neighbor14];
            meanvalue = nanmean(neighbors);
            aMap1(i,j) = meanvalue;
        end

    end
end

aMap1 = aMap1.*mask3;
aMap1(aMap1==0) = NaN;
aMap1 = round(aMap1);

cleanaMap = aMap1;
end