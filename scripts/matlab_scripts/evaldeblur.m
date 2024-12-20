%% Based on codes from https://github.com/swz30/MPRNet/blob/main/Deblurring/evaluate_GOPRO_HIDE.m
%% Evaluation by Matlab is often 0.01 better than Python for SSIM.
%% Euler command: module load matlab/R2020a; cd scripts/matlab_scripts; matlab -nodisplay -nojvm -singleCompThread -r evaluate_video_deblurring


close all;clear all;

datasets = {'BSD2'};
num_set = length(datasets);
file_paths = {'/Volumes/My Passport/CODE/mathlabSSIMCalData/results/BSD2_STFANetpp/visualization/*/'};
gt_paths = {'/Volumes/My Passport/CODE/mathlabSSIMCalData/BSD2_GT/*/'};

for idx_set = 1:num_set
    file_path = file_paths{idx_set};
    gt_path = gt_paths{idx_set};
    path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
    gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
    img_num = length(path_list);
    fprintf('For %s dataset, it has %d LQ images and %d GT images\n', datasets{idx_set}, length(path_list), length(gt_list));

    total_psnr = 0;
    total_ssim = 0;
    if img_num > 0
        for j = 1:img_num
           input = imread(strcat(path_list(j).folder, '/', path_list(j).name));
           gt = imread(strcat(gt_list(j).folder, '/', gt_list(j).name));
           ssim_val = ssim(input, gt);
           psnr_val = psnr(input, gt);
           fprintf('%s',psnr_val);
           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
           fprintf('%s,%s \n',strcat(path_list(j).folder, '/', path_list(j).name),strcat(gt_list(j).folder, '/', gt_list(j).name));
        end
       
    end
    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;

    fprintf('For %s dataset PSNR: %f SSIM: %f\n', datasets{idx_set}, qm_psnr, qm_ssim);

end