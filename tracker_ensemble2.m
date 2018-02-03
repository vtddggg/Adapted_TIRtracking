function [rects, time] = tracker_ensemble2(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, cell_size, show_visualization,ground_truth, enable_finetune)

% Input:
%   - video_path:          path to the image sequence
%   - img_files:           list of image names
%   - pos:                 intialized center position of the target in (row, col)
%   - target_sz:           intialized target size in (Height, Width)
% 	- padding:             padding parameter for the search area
%   - lambda:              regularization term for ridge regression
%   - output_sigma_factor: spatial bandwidth for the Gaussian label
%   - interp_factor:       learning rate for model update
%   - cell_size:           spatial quantization level
%   - show_visualization:  set to True for showing intermediate results
%   - ground_truth£º       use for visualize
%   - enable_finetune:     controlled if enable TIR finetuning
% Output:
%   - rects:           predicted target position at each frame
%   - time:                time spent for tracking
% ================================================================================
% Environment setting
% ================================================================================
npart = 4;
nbins = 16;
feat_width = 5; 
feat_sig = 0.625; 
sp_width = [9, 15];
sp_sig = [1, 2];
sp_weight = [1,1];
sp_weight  = reshape(sp_weight,1,1,[]);


init_rect = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
indLayers = [37, 28, 19];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
nweights  = [1, 0.5, 0.25]; % Weights for combining correlation filter responses
numLayers = length(indLayers);

% Get image size and search window size
im_sz     = size(imread([video_path img_files{1}]));
window_sz = get_search_window(target_sz, im_sz, padding);

% Compute the sigma for the Gaussian function label
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;

l1_patch_num = floor(window_sz/ cell_size);

% Pre-compute the Fourier Transform of the Gaussian function label
yf = fft2(gaussian_shaped_labels(output_sigma, l1_patch_num));

% Pre-compute and cache the cosine window (for avoiding boundary discontinuity)
cos_window = hann(size(yf,1)) * hann(size(yf,2))';

% Create video interface for visualization
if(show_visualization)
    update_visualization = show_video(img_files, video_path);
end

% Initialize variables for calculating FPS and distance precision
time      = 0;
rects = zeros(numel(img_files), 4);
nweights  = reshape(nweights,1,1,[]);

% Note: variables ending with 'f' are in the Fourier domain.
model_xf     = cell(1, numLayers);
model_alphaf = cell(1, numLayers);

p_model_xf = cell(npart,length(sp_width));
p_model_alphaf = cell(npart,length(sp_width));

annotations = get_part_rect(init_rect,npart);
part_pos = cell(npart,1);
part_sz = cell(npart,1);
part_w_sz = cell(npart,1);
confidence = ones(npart,1);
threshold = 0.1;
regular_sigma = 5;
    
    for i = 1:npart
        part_rect = annotations{i};
        part_pos{i} = part_rect([2,1])+part_rect([4,3])/2;
        part_sz{i} = part_rect([4,3]);
        part_w_sz{i} = floor(get_search_window(part_sz{i}, im_sz, padding));
    end
    pyf = fft2(gaussian_shaped_labels(sqrt(prod(part_sz{1})) * output_sigma_factor, part_w_sz{1}));
    pcos_window = hann(size(pyf,1)) * hann(size(pyf,2))';
response = zeros(part_w_sz{1}(1),part_w_sz{1}(2),npart);

current_scale_factor=1;

% ================================================================================
% Start tracking
% ================================================================================
for frame = 1:numel(img_files),
    im = imread([video_path img_files{frame}]); % Load the image at the current frame
    if ismatrix(im)
        im = cat(3, im, im, im);
    end
    
    tic();
    % ================================================================================
    % Predicting the object position from the learned object model
    % ================================================================================
    if frame > 1
         % Extracting hierarchical convolutional features
         feat = extractFeature(im, pos, window_sz, cos_window, indLayers);
         % Predict position
          pos  = predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
              model_xf, model_alphaf);
         % Scale estimation
          current_scale_factor = estimate_scale( rgb2gray(im), pos, current_scale_factor);   
          
      if enable_finetune == true
          target_sz_t=target_sz*current_scale_factor;
          box = [pos - target_sz_t/2, target_sz_t];
          part_pos = update_part_rect( box, part_pos,npart );
          
        for i = 1:npart
            part_im =  rgb2gray(get_subwindow(im, part_pos{i}, floor(part_w_sz{i}*current_scale_factor)));
            part_im = imresize(part_im,part_w_sz{i});
            df = img2df(part_im, nbins); %df feature
            res_layer = zeros([part_w_sz{i}, length(sp_width)]);
                for j=1:length(sp_width)
                    sdf = smoothDF(df, [sp_width(j) feat_width], [sp_sig(j), feat_sig]);
                    feature = bsxfun(@times, sdf, pcos_window);
                    zf = fft2(feature);
                    kzf=sum(zf .* conj(p_model_xf{i,j}), 3) / numel(zf);
                    temp= real(fftshift(ifft2(p_model_alphaf{i,j} .* kzf)));
                    res_layer(:,:,j)=temp/max(temp(:));
                end
                response(:,:,i) = sum(bsxfun(@times, res_layer, sp_weight), 3);
                % calcute confidence
                if frame >2
                    PSR = (max(response(:,i))-mean(response(:,i)))/std(response(:,i));
                    res = response(:,:,i);
                    [yy_shift, xx_shift] = find(res == max(res(:)), 1);
                    yy_shift = yy_shift - floor(size(zf,1)/2);
                    xx_shift = xx_shift - floor(size(zf,2)/2);
                    GR = exp(-(yy_shift.^2+xx_shift.^2)/(2*regular_sigma.^2));
                    confidence(i) = GR * PSR;
                end
                res = response(:,:,i);
                [vert_delta, horiz_delta] = find(res == max(res(:)), 1);
                vert_delta  = vert_delta  - floor(size(zf,1)/2);
                horiz_delta = horiz_delta - floor(size(zf,2)/2);
                part_pos{i} = part_pos{i} + [vert_delta - 1, horiz_delta - 1];
                
                % update annotations
                annotations{i} = [part_pos{i}([2,1])-part_sz{i}([2,1])*current_scale_factor/2,part_sz{i}([2,1])*current_scale_factor];                
        end
        % assemble
        confidence(confidence<threshold) = 0;
        if sum(confidence.^2) ~= 0
            confidence = confidence/sqrt(sum(confidence.^2));
            response_final = sum(bsxfun(@times, response, reshape(confidence,1,1,[])), 3);
            [vert_delta, horiz_delta] = find(response_final == max(response_final(:)), 1);
            vert_delta  = vert_delta  - floor(size(zf,1)/2);
            horiz_delta = horiz_delta - floor(size(zf,2)/2);
            pos = pos + [vert_delta - 1, horiz_delta - 1];
        end
      end
        

    else
        init_scale_para(rgb2gray(im), target_sz, pos);
    end
    
    % Extracting convolutional features
     feat  = extractFeature(im, pos, window_sz, cos_window, indLayers);
     % Model update
     [model_xf, model_alphaf] = updateModel(feat, yf, interp_factor, lambda, frame, ...
         model_xf, model_alphaf);
   if enable_finetune == true
    for i = 1:npart
        feature = cell(1,length(sp_width));
        part_im =  rgb2gray(get_subwindow(im, part_pos{i}, floor(part_w_sz{i}*current_scale_factor)));
        part_im = imresize(part_im,part_w_sz{i});
        df = img2df(part_im, nbins); %df feature
        for j=1:length(sp_width)
            sdf = smoothDF(df, [sp_width(j) feat_width], [sp_sig(j), feat_sig]);
            feature{j} = bsxfun(@times, sdf, pcos_window);
        end
        if confidence(i) > threshold
            % update 
            [p_model_xf(i,:), p_model_alphaf(i,:)] = updateModel(feature, pyf, interp_factor*confidence(i), lambda, frame, ...
    p_model_xf(i,:), p_model_alphaf(i,:));
        end
    end
   end
    
    
    target_sz_t=target_sz*current_scale_factor;
    box = [pos([2,1]) - target_sz_t([2,1])/2, target_sz_t([2,1])];
    box(2,:) = [ground_truth(frame,1)-ground_truth(frame,3)/2 ground_truth(frame,2)-ground_truth(frame,4)/2 ground_truth(frame,3) ground_truth(frame,4)];    
    rects(frame,:)=box(1,:);

    time = time + toc();
    
    % Visualization
    if show_visualization,
        %box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        drawnow
        % 			pause(0.05)  % uncomment to run slower
    end
end

end



function pos = predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
    model_xf, model_alphaf)

% ================================================================================
% Compute correlation filter responses at each layer
% ================================================================================
res_layer = zeros([l1_patch_num, length(indLayers)]);

for ii = 1 : length(indLayers)
    zf = fft2(feat{ii});
    kzf=sum(zf .* conj(model_xf{ii}), 3) / numel(zf);
    
    temp= real(fftshift(ifft2(model_alphaf{ii} .* kzf)));  %equation for fast detection
    res_layer(:,:,ii)=temp/max(temp(:));
end

% Combine responses from multiple layers (see Eqn. 5)
response = sum(bsxfun(@times, res_layer, nweights), 3);

% ================================================================================
% Find target location
% ================================================================================
% Target location is at the maximum response. we must take into
% account the fact that, if the target doesn't move, the peak
% will appear at the top-left corner, not at the center (this is
% discussed in the KCF paper). The responses wrap around cyclically.
[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
vert_delta  = vert_delta  - floor(size(zf,1)/2);
horiz_delta = horiz_delta - floor(size(zf,2)/2);

% Map the position to the image space
pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];


end


function [model_xf, model_alphaf] = updateModel(feat, yf, interp_factor, lambda, frame, ...
    model_xf, model_alphaf)

numLayers = length(feat);

% ================================================================================
% Initialization
% ================================================================================
xf       = cell(1, numLayers);
alphaf   = cell(1, numLayers);

% ================================================================================
% Model update
% ================================================================================
for ii=1 : numLayers
    xf{ii} = fft2(feat{ii});
    kf = sum(xf{ii} .* conj(xf{ii}), 3) / numel(xf{ii});
    alphaf{ii} = yf./ (kf+ lambda);   % Fast training
end

% Model initialization or update
if frame == 1,  % First frame, train with a single image
    for ii=1:numLayers
        model_alphaf{ii} = alphaf{ii};
        model_xf{ii} = xf{ii};
    end
else
    % Online model update using learning rate interp_factor
    for ii=1:numLayers
        model_alphaf{ii} = (1 - interp_factor) * model_alphaf{ii} + interp_factor * alphaf{ii};
        model_xf{ii}     = (1 - interp_factor) * model_xf{ii}     + interp_factor * xf{ii};
    end
end


end

function feat  = extractFeature(im, pos, window_sz, cos_window, indLayers)

% Get the search window from previous detection
patch = get_subwindow(im, pos, window_sz);
% Extracting convolutional features
feat  = get_features(patch, cos_window, indLayers);

end

