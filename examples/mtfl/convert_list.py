__author__ = 'ytoon'

import h5py
hdf5_path = '/media/ytoon/Elements/mtfl/train_data/train2.h5'

count = 0
with open('training.txt', 'r') as f:
    for line in f.readlines():
        count = count + 1
print count
# with open('/media/ytoon/Elements/mtfl/testing_list.txt', 'r') as f:
#     with open('testing.txt', 'w+') as fid:
#         for line in f.readlines():
#             fid.write(line.split('/')[1])

f = h5py.File(hdf5_path, 'r')

print 'Done.'