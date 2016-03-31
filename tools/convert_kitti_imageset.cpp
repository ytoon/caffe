#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/blob.hpp"

#include <cstdio>

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
// DEFINE_bool(check_size, false,
//     "When this option is on, check that all the datum have the same size");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images by subtract mean image\n"
        "Usage:\n"
        "    convert_kitti_imageset [FLAGS] ROOTFOLDER/ LISTFILE NEWFOLDER MEANFILE/\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_kitti_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const unsigned int height = 370;
  const unsigned int width = 1226;
  // const bool check_size = FLAGS_check_size;
  std::string new_folder(argv[3]);
  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  // printf("%d\n", static_cast<int>(lines.size()));
  LOG(INFO) << "A total of " << lines.size() << " images.";
  // Load mean image
  std::string mean_file(argv[4]);
  LOG(INFO) << "Loading mean file from: " << mean_file;
  BlobProto blob_proto;
  Blob<double> data_mean;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  data_mean.FromProto(blob_proto);

  const int mean_width = data_mean.width();
  const int mean_height = data_mean.height();

  const double* mean = data_mean.cpu_data();

  std::string root_folder(argv[1]);
  int count = 0;

  const int channels = 3;
  cv::Mat mean_image(height, width, CV_64FC3);
  printf("%d\n", mean_image.rows);
  printf("%d\n", mean_image.cols);
  printf("%d\n", mean_image.channels());

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    cv::Mat im = ReadImageToCVMat(root_folder + lines[line_id].first,
        height, width, is_color);

    for (int h = 0; h < im.rows; ++h) {
      const uchar* ptr = im.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < im.cols; ++w) {
        for (int c = 0; c < channels; ++c) {
          double pixel = static_cast<double>(ptr[img_index++]);

          int mean_index = (c * mean_height + h) * mean_width + w;
          // printf("%d\n", mean_image.rows);
          // TODO: segmentation error
          mean_image.at<double>(h, w, c) = pixel - mean[mean_index];
        }
      }
    }
    std::string path = "./test.png";
    cv::imwrite(path, mean_image);

    if (++count % 1000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
