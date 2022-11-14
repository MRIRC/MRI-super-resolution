function y = HPF(image)
kernel1 = -1* ones(3)/(9);
kernel1(2,2) = 8/9;
kernel2 = [-1 -2 -1; -2 12 -2; -1 -2 -1]/16;
H = fspecial('unsharp');
% Filter the image.  Need to cast to single so it can be floating point
% which allows the image to have negative values.
y = imfilter(single(image), H);
end