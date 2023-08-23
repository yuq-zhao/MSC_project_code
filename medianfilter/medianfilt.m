clear,clc;
pauseTime = 0;

results = table('Size', [0 4], 'VariableTypes', {'string', 'double', 'double', 'double'}, 'VariableNames', {'ImageName', 'NoiseLevel', 'PSNR', 'SSIM'});


data_path = "..\Set12";
ext = ["*.jpg", "*.png", "*.jpeg"];
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(data_path,ext(i))));
end
noise_leval = [10,15,20,25,30,35,40,45,50,55,60,65,70];

for ii = 1:length(noise_leval)
    PSNRs = zeros(1, length(filePaths));
    SSIMs = zeros(1, length(filePaths));
    sigma = noise_leval(ii);
    for jj = 1:length(filePaths)
        % ԭͼ��
        originImage = im2double(imread(filePaths(jj).name));
        % ��Ӹ�˹����
        imageWithNoise = single(originImage + sigma/255*randn(size(originImage)));
        [rows, cols] = size(originImage);
        y = imageWithNoise;

        % ��ֵ�˲��㷨
        % ָ��ģ��ߴ�
        boxSize = 3;
        template = zeros(boxSize);
        for i = 1:rows-boxSize+1
            for j = 1:cols-boxSize+1
                % ȡģ��������
                template = imageWithNoise(i:i+(boxSize-1),j:j+(boxSize-1));
                % ����ֵ�滻ģ�����ĵ�����ֵ
                m = median(template(:));
                y(i+(boxSize-1)/2,j+(boxSize-1)/2) = m;
            end
        end

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
writetable(results,  'results.csv');

