% Fix existing masks. As in make some pixels 0 if they are noisy.
function [mask3] = maskfix

% First load existing mask
mask3 = load('C:\Users\Sofia\Desktop\Optical data - Mice\Heart #2\mask3.txt');

% Visualize existing mask if you want
% figure
% imagesc(mask3)

% Select pixels to turn to 0
% for x = 87:230
%     for y = 10:151
%         mask3(x,y) = 0;
%     end
% end

% Visualize final mask if you want
% figure
% imagesc(mask3)
end
