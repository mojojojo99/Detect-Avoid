a=VideoReader('/Users/jolim/Desktop/trimmed_v3.mp4');

for img = 1:a.NumberOfFrames;
    filename=strcat('frame',num2str(img),'.jpg');
    b = read(a, img);
    imshow(b);
    imwrite(b,filename);
end
movie(img)