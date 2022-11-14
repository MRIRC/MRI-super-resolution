clc
clear
close all

start_indices = [751, 1964, 4390, 3177];
diff = 1761-751;
SSIM = [];
FSIM_ = [];
SR_SIM_ = [];
MULTISSIM = [];
PSNR = [];


HF_power = [];

[~, ~, labels] = xlsread('labels.csv');

for row = 2: 66%size(labels,1)
    row
    filename = [num2str(labels{row, 2}), '.png'];
    cdata = imread(filename);


    x = rgb2gray(cdata(381:1390,:,:));
    for ii = 1:4
        index = start_indices(ii);
        x1 = x(:,index:index+diff);
        x1 = x1(300:700,300:700);

        if strcmp(labels{row, ii+4}, 'base')
            HR = x1;
        elseif strcmp(labels{row, ii+4}, 'interpolated')
            inter = x1;
        elseif strcmp(labels{row, ii+4}, 'low')
            LR = x1;
        else
            SR = x1;
        end
    end

    pow_HR = sum(sum(HPF(HR).^2));
    pow_inter = sum(sum(HPF(inter).^2));
    power_diff = max(HPF(SR) - HPF(inter),0);
    power_diff = sum(sum(power_diff.^2));
    pow_SR = mean(mean(HPF(SR).^2));

    HF_power = [HF_power, power_diff/pow_inter];
    

    SSIM = [SSIM; [ssim(inter,HR), ssim(SR, HR), ssim(HPF(inter), HPF(HR)), ssim(HPF(SR), HPF(HR))]];
    PSNR = [PSNR; [immse(inter,HR), immse(SR, HR), immse(HPF(inter), HPF(HR)), immse(HPF(SR), HPF(HR))]];
    MULTISSIM = [MULTISSIM; [multissim(inter,HR), multissim(SR, HR), multissim(HPF(inter), HPF(HR)), multissim(HPF(SR), HPF(HR))]];
    FSIM_ = [FSIM_; [FSIM(inter,HR), FSIM(SR, HR), FSIM(HPF(inter), HPF(HR)), FSIM(HPF(SR), HPF(HR))]];
    SR_SIM_ = [SR_SIM_; [SR_SIM(inter,HR), SR_SIM(SR, HR), SR_SIM(HPF(inter), HPF(HR)), SR_SIM(HPF(SR), HPF(HR))]];

%     disp('increase')
%     disp(100*(ssim((SR),(HR))-ssim((inter),(HR)))/ssim((inter),(HR)))
%     disp(100*(ssim(HPF(SR),HPF(HR))-ssim(HPF(inter),HPF(HR)))/ssim(HPF(inter),HPF(HR)))
%     disp('power')
%     disp(100*(power_diff/pow_inter))
end

clc
close all
stat = SSIM;
%mean(stat)
%std(stat)
[~, p] = ttest(stat(:,1),stat(:,2));
figure
subplot(221)
x = stat(:,3);
y = stat(:,4);
plot(x, y, '.', "MarkerSize", 15)
hold on 
plot(x,x, 'k-')
title("SSIM Index")
legend(["(bicubic, SR)", "x=y line"], "Location","northwest")



stat = MULTISSIM;
%mean(stat)
%std(stat)
[~, p] = ttest(stat(:,1),stat(:,2));
subplot(222)
x = stat(:,3);
y = stat(:,4);
plot(x, y, '.', "MarkerSize", 15)
hold on 
plot(x,x, 'k-')
title("MS-SSIM Index")
legend(["(bicubic, SR)", "x=y line"], "Location","northwest")



stat = SR_SIM_;
%mean(stat)
%std(stat)
[~, p] = ttest(stat(:,1),stat(:,2));
subplot(223)
x = stat(:,3);
y = stat(:,4);
plot(x, y, '.', "MarkerSize", 15)
hold on 
plot(x,x, 'k-')
title("SR-SSIM Index")
legend(["(bicubic, SR)", "x=y line"], "Location","northwest")

stat = FSIM_;
%mean(stat)
%std(stat)
[~, p] = ttest(stat(:,1),stat(:,2));
subplot(224)
x = stat(:,3);
y = stat(:,4);
plot(x, y, '.', "MarkerSize", 15)
hold on 
plot(x,x, 'k-')
title("FSIM Index")
legend(["(bicubic, SR)", "x=y line"], "Location","northwest")


