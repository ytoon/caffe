__author__ = 'ytoon'


import caffe
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
hdf5_path = '/media/ytoon/Elements/mtfl/train_data/train1.h5'

caffe_root = '/home/ytoon/caffe-master/'
image_path = '/media/ytoon/Elements/mtfl/dataset_test/_60_930_0.jpg'
net = caffe.Net(caffe_root + 'models/mtfl/deploy.prototxt',
                caffe_root + 'models/mtfl/model_full_landmark/caffe_mtflnet_train_iter_30000.caffemodel',
                caffe.TEST)

net.blobs['data'].reshape(1,1,40,40)

im = Image.open(image_path)
im = im.resize((40, 40), Image.BICUBIC)
# plt.imshow(im)
# plt.show()
im_gray = im.convert('L')
in_ = np.asarray(im_gray) / 255.
net.blobs['data'].data[...] = in_

# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_raw_scale('data', 255)

# net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path, color=False))

caffe.set_device(0)
caffe.set_mode_gpu()

out = net.forward()

xy = []
for i in range(0, 5):
    xy.append(out['fc_landmark'][0][i] * 40)
    xy.append(out['fc_landmark'][0][i + 5] * 40)

print out['glasses']

draw = ImageDraw.Draw(im)
draw.point(xy, fill=(0, 255, 0))
plt.imshow(im)
plt.show()
print "Done."
