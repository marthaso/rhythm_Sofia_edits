% Fix existing masks. As in make some pixels 0 if they are noisy.
% Toggle the value of the clicked pixel
    
function [mask3] = maskfix

mask3 = uigetfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\mask3_new.txt');
mask3 = load(mask3);

while(1)

figure
imagesc(mask3)

h = drawfreehand;
newmask = createMask(h);
% figure
% imagesc(newmask)

new_mask = mask3-newmask;
new_mask(new_mask==-1)=0;
imagesc(new_mask)

add = drawfreehand;
newmaskplus = createMask(add);
new_maskplus = new_mask + newmaskplus;
new_maskplus(new_maskplus==2)=1;
% figure
% imagesc(new_maskplus)
mask3 = new_maskplus;


choice = menu('Continue editing?','Yes','No');
if choice==2 || choice==0
   close figure
   prompt = "Save data as... ";
   filename = input(prompt);
   save(filename, 'mask3','-ascii', '-tabs');
   break;
end
end




% 
% 
% % First load existing mask
% mask3 = load('C:\Users\Sofia\Desktop\Optical data - Mice\Heart #2\mask3.txt');
% 
% %mask3 = ones(20,20);
% 
% % Visualize existing mask if you want
% figure
% imagesc(mask3);
% axis on
% %grid on
% 
% %[x, y] = getpts;
% % poly_indices = convhull(x, y);
% % 
% % [x,y,P] = impixel(mask3);
% x = [51, 51, 51, 50, 50, 50, 50];
% y=[96,95,94,94,95,96,97];
% %y = [160;161;162;161];
% for i = 1:length(x)-1
%         mask3(y(i),x(i)) = 0;
% end
% 
% 
% figure
% imagesc(mask3)
% grid on
% 
% save('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\mask3_new2.txt','mask3','-ascii', '-tabs')
% %[x,y] = getpts
% 
% b=1;
end
    


% Select pixels to turn to 0
% for x = 87:230
%     for y = 10:151
%         mask3(x,y) = 0;
%     end
% end

% Visualize final mask if you want
% figure
% imagesc(mask3)

