clc
clear
close
load('pt47_super.mat')
load('pt47_slice13.mat')

figure('Renderer', 'painters', 'Position', [10 10 1500 1500])
pause (0.5)
subplot(1,3,1)
mean_img = mean(data_fixed_p_s,3);
imagesc(mean_img)
colormap("gray")
axis square off
title('Mean image')
set(gca,'Fontsize',16)
subplot(1,3,2)

for i=1:40
    imagesc(epochs(:,:,i))
    colormap("gray")
    axis square off
    title(['Reconstruction after epoch ',num2str((i-1)*25)])
    set(gca,'Fontsize',16)
    pause (0.05)
end
subplot(1,3,3)
imagesc(mean(data_fixed_p_s,3))
colormap("gray")
axis square off
title('Mean of Reconstructions')
set(gca,'Fontsize',16)

figure('Renderer', 'painters', 'Position', [10 10 1500 1500])

subplot(121)
imagesc(mean(epochs(:,:,2:end),3))
title('Mean of every 25 epochs')
colormap(gray)
axis square off
subplot(122)
imagesc(imresize(mean_img,4))
title('mean image interpolated')
axis square off

figure('Renderer', 'painters', 'Position', [10 10 1500 1500])

subplot(121)
imagesc(epochs(:,:,end))
title('After Epoch 950')
axis square off
subplot(122)
imagesc(imresize(mean_img,4))
title('mean image interpolated')
axis square off
colormap(gray)