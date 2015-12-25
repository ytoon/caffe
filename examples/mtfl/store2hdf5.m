function [curr_dat_sz] = store2hdf5(filename, data, label, attributes, create, startloc, chunksz)  
  % *data* is W*H*C*N matrix of images should be normalized (e.g. to lie between 0 and 1) beforehand
  % *label* is N*D matrix of labels (D axis(x&y) per sample) 
  % *create* [0/1] specifies whether to create file newly or to append to previously created file, useful to store information in batches when a dataset is too big to be held in memory  (default: 1)
  % *startloc* (point at which to start writing data). By default, 
  % if create=1 (create mode), startloc.data=[1 1 1 1], and startloc.lab=[1 1]; 
  % if create=0 (append mode), startloc.data=[1 1 1 K+1], and startloc.lab = [1 K+1]; where K is the current number of samples stored in the HDF
  % chunksz (used only in create mode), specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) for creating HDF5 files with unbounded maximum size - TLDR; higher chunk sizes allow faster read-write operations 

  % verify that format is right
  dat_dims=size(data);
  lab_dims=size(label);

  assert(lab_dims(end)==dat_dims(end), 'Number of samples should be matched between data and attr labels');

  if ~exist('create','var')
    create=true;
  end

  
  if create
    %fprintf('Creating dataset with %d samples\n', num_samples);
    if ~exist('chunksz', 'var')
      chunksz=1000;
    end
    if exist(filename, 'file')
      fprintf('Warning: replacing existing file %s \n', filename);
      delete(filename);
    end      
    h5create(filename, '/data', [dat_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [dat_dims(1:end-1) chunksz]); % width, height, channels, number 
    h5create(filename, '/landmark', [10 Inf], 'Datatype', 'single', 'ChunkSize', [10 chunksz]);
    h5create(filename, '/gender', [1 Inf], 'Datatype', 'single', 'ChunkSize', [1 chunksz]);
    h5create(filename, '/smile', [1 Inf], 'Datatype', 'single', 'ChunkSize', [1 chunksz]);
    h5create(filename, '/glasses', [1 Inf], 'Datatype', 'single', 'ChunkSize', [1 chunksz]);
    h5create(filename, '/pose', [1 Inf], 'Datatype', 'single', 'ChunkSize', [1 chunksz]);
    if ~exist('startloc','var') 
      startloc.data=[ones(1,length(dat_dims)-1), 1];
    end 
  else  % append mode
    if ~exist('startloc','var')
      info=h5info(filename);
      prev_dat_sz=info.Datasets(1).Dataspace.Size;
      assert(prev_dat_sz(1:end-1)==dat_dims(1:end-1), 'Data dimensions must match existing dimensions in dataset');
      
      startloc.dat=[ones(1,length(dat_dims)-1), prev_dat_sz(end)+1];
      % 
    end
  end

  if ~isempty(data)
    h5write(filename, '/data', single(data), startloc.data, size(data));
    h5write(filename, '/landmark', single(label), startloc.landmark, size(label));
    h5write(filename, '/gender', single(attributes(1,:)), startloc.gender, size(attributes(1,:)));
    h5write(filename, '/smile', single(attributes(2,:)), startloc.smile, size(attributes(2,:)));
    h5write(filename, '/glasses', single(attributes(3,:)), startloc.glasses, size(attributes(3,:)));
    h5write(filename, '/pose', single(attributes(4,:)), startloc.pose, size(attributes(4,:)));
  end

  if nargout
    info=h5info(filename);
    curr_dat_sz=info.Datasets(1).Dataspace.Size;
  end
end
