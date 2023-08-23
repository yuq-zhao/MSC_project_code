clear,clc;
pauseTime = 0;

data_path = "..\Set12";
ext = ["*.jpg", "*.png", "*.jpeg"];
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(data_path,ext(i))));
end
noise_leval = [10,15,20,25,30,35,40,45,50,55,60,65,70];

results = table('Size', [0 4], 'VariableTypes', {'string', 'double', 'double', 'double'}, 'VariableNames', {'ImageName', 'NoiseLevel', 'PSNR', 'SSIM'});

for ii = 1:length(noise_leval)
    PSNRs = zeros(1, length(filePaths));
    SSIMs = zeros(1, length(filePaths));
    sigma = noise_leval(ii);
    for jj = 1:length(filePaths)
        disp(['���ڴ���ͼƬ��', filePaths(jj).name]);
        % ԭͼ��
        originImage = im2double(imread(filePaths(jj).name));
        % ��Ӹ�˹����
        imageWithNoise = single(originImage + sigma/255*randn(size(originImage)));
%         imageWithNoise = imnoise(originImage,'gaussian',0,(sigma/255)^2);
        % NL-Means�˲�
%         y = NLmeans(imageWithNoise,2,7,sigma/100);
        y = imnlmfilt(imageWithNoise);
        % ����psnr��ssim
        PSNRs(jj) = psnr(im2uint8(originImage), im2uint8(y));
        SSIMs(jj) = ssim(im2uint8(originImage), im2uint8(y));
        combinedImage = cat(2,im2uint8(originImage),im2uint8(imageWithNoise),im2uint8(y));
        imshow(combinedImage);
        title(['sigma=',num2str(sigma),'  ',filePaths(jj).name,'  psnr=',num2str(PSNRs(jj),'%2.2f'),'dB','  ssim=',num2str(SSIMs(jj),'%2.4f')])
        drawnow;
        % ����ͼ�񵽵�ǰ�����ļ���
        imwrite(combinedImage, sprintf('Processed_%s_Sigma%d.png', filePaths(jj).name, sigma));
        results(end+1, :) = {filePaths(jj).name, sigma, PSNRs(jj), SSIMs(jj)};
        pause(pauseTime)
    end
    disp(["sigma:",sigma,"psnr:",mean(PSNRs),"ssim:", mean(SSIMs)]);
end
% �������ݵ�csv�ļ�
writetable(results,  'results.csv');
