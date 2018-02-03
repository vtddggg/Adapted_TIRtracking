
% Coding by Xiaofeng Mao (hzmaoxiaofeng@163.com or mmmaoxiaofeng@gmail.com) 
% This TIR-tracking finetuning demo is based on Hierarchical Convolutional Features for Visual Tracking
% Citation:
% @inproceedings{Ma-ICCV-2015,
%    title={Hierarchical Convolutional Features for Visual Tracking},
%    Author = {Ma, Chao and Huang, Jia-Bin and Yang, Xiaokang and Yang, Ming-Hsuan},
%    booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
%    pages={},
%    Year = {2015}
%    }

addpath('utility','model','scale','external/matconvnet/matlab');
vl_setupnn();

% Note that the default setting does not enable GPU
% TO ENABLE GPU, recompile the MatConvNet toolbox 
global enableGPU;
enableGPU = false;

show_visualization = true;
show_precision = true;
enable_finetune = false;


run_tracker('crowd', show_visualization, show_precision, enable_finetune);