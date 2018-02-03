% RUN_TRACKER: process a specified video using CF2
%
% Input:
%     - video:              the name of the selected video
%     - show_visualization: set to True for visualizing tracking results
%     - show_plots:         set to True for plotting quantitative results
% Output:
%     - precision:          precision thresholded at 20 pixels
%
%
function [precision, fps] = run_tracker(video, show_visualization, show_plots, enable_finetune)

base_path   = './data';

% Extra area surrounding the target
padding = struct('generic', 1.8, 'large', 1, 'height', 0.4);

lambda = 1e-4;              % Regularization parameter (see Eqn 3 in our paper)
output_sigma_factor = 0.1;  % Spatial bandwidth (proportional to the target size)

interp_factor = 0.01;       % Model learning rate (see Eqn 6a, 6b)
cell_size = 4;              % Spatial cell size

[img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);
[rects, time] = tracker_ensemble2(video_path, img_files, pos, target_sz, ...
            padding, lambda, output_sigma_factor, interp_factor, ...
            cell_size, show_visualization,ground_truth, enable_finetune);
% Calculate and show precision plot, as well as frames-per-second
positions=rects(:,[1 2])+rects(:,[3 4])/2;
precisions = precision_plot(positions, ground_truth, video, show_plots);
fps = numel(img_files) / time;
        
fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)
        
if nargout > 0,
    %return precisions at a 20 pixels threshold
    precision = precisions(20);
end
end
