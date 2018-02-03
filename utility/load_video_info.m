function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


	%see if there's a suffix, specifying one of multiple targets, for
	%example the dot and number in 'Jogging.1' or 'Jogging.2'.
	if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end))),
		suffix = video(end-1:end);  %remember the suffix
		video = video(1:end-2);  %remove it from the video name
	else
		suffix = '';
	end

	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
	end
	video_path = [base_path video '/'];

	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path 'groundtruth_rect' suffix '.txt'];
    data = dlmread(filename);
    ground_truth = zeros(size(data,1),4);
if size(data,2) == 8
    ground_truth(:,1) = mean(data(:,1:2:end),2);
    ground_truth(:,2) = mean(data(:,2:2:end),2);
    x1 = min(data(:,1:2:end),[],2);
    x2 = max(data(:,1:2:end),[],2);
    y1 = min(data(:,2:2:end),[],2);
    y2 = max(data(:,2:2:end),[],2);
    A1 = sqrt(sum((data(:,1:2) - data(:,3:4)).^2,2)) .* sqrt(sum((data(:,3:4) - data(:,5:6)).^2,2));
    A2 = (x2 - x1) .* (y2 - y1);
    s = sqrt(A1./A2);
    ground_truth(:,3) = s .* (x2 - x1) + 1;
    ground_truth(:,4) = s .* (y2 - y1) + 1;
    else
    ground_truth = data;
end
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];
	pos = [ground_truth(1,2), ground_truth(1,1)];
	
	if size(ground_truth,1) == 1,
		%we have ground truth for the first frame only (initial position)
		ground_truth = [];
	else
		%store positions instead of boxes
		%ground_truth = ground_truth(:,[2,1]);
	end
	
	
	%from now on, work in the subfolder where all the images are
	
	%for these sequences, we must limit ourselves to a range of frames.
	%for all others, we just load all png/jpg files in the folder.
	frames = {'David', 300, 770;
			  'Football1', 1, 74;
			  'Freeman3', 1, 460;
			  'Freeman4', 1, 283;
              'Diving', 1, 215};
	
	idx = find(strcmpi(video, frames(:,1)));
	
	if isempty(idx),
		%general case, just list all images
		img_files = dir([video_path '*.png']);
		if isempty(img_files),
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
		end
		img_files = sort({img_files.name});
	else
		%list specified frames. try png first, then jpg.
		if exist(sprintf('%s%08i.png', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%08i.png');
			
		elseif exist(sprintf('%s%08i.jpg', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%08i.jpg');
			
		else
			error('No image files to load.')
		end
		
		img_files = cellstr(img_files);
	end
	
end

