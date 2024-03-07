load('C:\Users\Sofia\Desktop\Optical data - Mice\Heart #2\mask2.txt')

mask2(130,65)=0;

save('C:\Users\Sofia\Desktop\Optical data - Mice\Heart #2\mask3.txt','mask2','-ascii', '-tabs')