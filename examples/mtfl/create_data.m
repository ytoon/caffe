%% WRITING TO HDF5
fileprex='/media/ytoon/Elements/mtfl/train_data/train';
filetype='.h5';
filename='/media/ytoon/Elements/mtfl/train_data/train1.h5';
image_data_path = '/media/ytoon/Elements/mtfl/dataset/';

% open the file and read data to a matrix
file_id = fopen('training.txt', 'r');

% CREATE list.txt containing filename, to be used as source for HDF5_DATA_LAYER
FILE=fopen('train_list.txt', 'w');

if file_id == -1
    fprintf('no such file');
    exit(-1);
end

num_total_samples = 10000;

chunksz=100;
created_flag=false;
totalct=0;
count = 2;

for batchno=1:num_total_samples/chunksz
  fprintf('batch no. %d\n', batchno);
  batch_data = [];
  batch_landmark = [];
  batch_attributes = [];
  
  % to read data to be held in memory before dumping to hdf5 file
  for i = 1:chunksz
      txt = strsplit(fgets(file_id));
      txt = txt(1:15);
      im_path = strcat(image_data_path, txt(1, 1));
      im = imread(char(im_path));
      [h, w, ~] = size(im);
      im = imresize(im, [40, 40]);
      [~, ~, c] = size(im);
      if c == 3
          im = rgb2gray(im);
      end
      im = im2double(im);
      % subtract mean image
%       I(:, :, 3) = (im(:, :, 1) - im_mean(:, :, 3))';
%       I(:, :, 2) = (im(:, :, 2) - im_mean(:, :, 2))';
%       I(:, :, 1) = (im(:, :, 3) - im_mean(:, :, 1))';
      batch_data(:, :, 1, i) = im';
      landmark = cellfun(@str2num, txt(2:11));
      landmark(1:5) = landmark(1:5) / w;
      landmark(6:10) = landmark(6:10) / h;
      batch_landmark(:, i) = landmark;
      batch_attributes(:, i) = cellfun(@str2num, txt(12:15)) - 1;
  end
  % store to hdf5
  startloc=struct('data', [1,1,1,totalct+1], 'landmark', [1,totalct+1], 'gender', [1,totalct+1], 'smile', [1,totalct+1], 'glasses', [1,totalct+1], 'pose', [1,totalct+1]);
  curr_dat_sz=store2hdf5(filename, batch_data, batch_landmark, batch_attributes, ~created_flag, startloc, chunksz); 
  created_flag=true;% flag set so that file is created only once
  totalct=curr_dat_sz(end);% updated dataset size (#samples)
  if mod(batchno, 100) == 0
      % display structure of the stored HDF5 file
      h5disp(filename);
      fprintf(FILE, '%s\n', filename);
      filename = strcat(fileprex, num2str(count), filetype);
      created_flag = false;
      totalct = 0;
      count = count + 1;
  end
end

% h5disp(filename);
fprintf(FILE, '%s\n', filename);

% close the open file
fclose('all');

fprintf('HDF5 filename listed in %s \n', 'train_list.txt');

% NOTE: In net definition prototxt, use list.txt as input to HDF5_DATA as: 
% layer {
%   name: "data"
%   type: "HDF5Data"
%   top: "data"
%   top: "labelvec"
%   hdf5_data_param {
%     source: "/path/to/list.txt"
%     batch_size: 64
%   }
% }