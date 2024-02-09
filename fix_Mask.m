load('C:\Users\Sofia\Desktop\Optical data - Mice\Heart #3 (WT)\mask2.txt')

mask2(124,42)=0;

save('C:\Users\Sofia\Desktop\Optical data - Mice\Heart #3 (WT)\mask3.txt','mask2','-ascii', '-tabs')