#include <vector>

#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  LocalConvolutionParameter local_conv_param = this->layer_param_.local_convolution_param();
  this->force_nd_im2col_ = local_conv_param.force_nd_im2col();
  this->channel_axis_ = bottom[0]->CanonicalAxisIndex(local_conv_param.axis());
  const int first_spatial_axis = this->channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  this->num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(this->num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  if (local_conv_param.has_kernel_h() || local_conv_param.has_kernel_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, local_conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = local_conv_param.kernel_h();
    kernel_shape_data[1] = local_conv_param.kernel_w();
  } else {
    const int num_kernel_dims = local_conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims);";
      for (int i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            local_conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = this->stride_.mutable_cpu_data();
  if (local_conv_param.has_stride_h() || local_conv_param.has_stride_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, local_conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = local_conv_param.stride_h();
    stride_data[1] = local_conv_param.stride_w();
  } else {
    const int num_stride_dims = local_conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          local_conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = this->pad_.mutable_cpu_data();
  if (local_conv_param.has_pad_h() || local_conv_param.has_pad_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, local_conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = local_conv_param.pad_h();
    pad_data[1] = local_conv_param.pad_w();
  } else {
    const int num_pad_dims = local_conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          local_conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup kernel_share dimensions (weight_share_).
  kernel_share_.Reshape(spatial_dim_blob_shape);
  int* kernel_share_data = kernel_share_.mutable_cpu_data();
  if (local_conv_param.has_kernel_share_h() || local_conv_param.has_kernel_share_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "kernel_share_h & kernel_share_w can only be used for 2D convolution.";
    CHECK_EQ(0, local_conv_param.kernel_share_size_size())
        << "Either kernel_share_size or kernel_share_h/w should be specified; not both.";
    kernel_share_data[0] = local_conv_param.kernel_share_h();
    kernel_share_data[1] = local_conv_param.kernel_share_w();
  } else {
    const int num_kernel_share_dims = local_conv_param.kernel_share_size_size();
    CHECK(num_kernel_share_dims == 0 || num_kernel_share_dims == 1 ||
          num_kernel_share_dims == this->num_spatial_axes_)
        << "kernel_share_size must be specified once, or once per spatial dimension "
        << "(kernel_share_size specified " << num_kernel_share_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      kernel_share_data[i] = (num_kernel_share_dims == 0) ? kDefaultPad :
          local_conv_param.kernel_share_size((num_kernel_share_dims == 1) ? 0 : i);
    }
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    this->is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!this->is_1x1_) { break; }
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(this->channel_axis_);
  this->num_output_ = this->layer_param_.local_convolution_param().num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.local_convolution_param().group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    this->set_conv_out_channels(this->channels_);
    this->set_conv_in_channels(this->num_output_);
  } else {
    this->set_conv_out_channels(this->num_output_);
    this->set_conv_in_channels(this->channels_);
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = this->num_output_ * kernel_share_data[0] * kernel_share_data[1];
  weight_shape[1] = this->channels_ / this->group_;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  this->bias_term_ = this->layer_param_.local_convolution_param().bias_term();
  vector<int> bias_shape(this->bias_term_, this->num_output_ * kernel_share_data[0] * kernel_share_data[1]);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.local_convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.local_convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->set_kernel_dim(this->blobs_[0]->count(1));
  this->weight_offset_ = this->num_output_ * kernel_share_data[0] * kernel_share_data[1] * this->blobs_[0]->count(1) / this->group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = this->channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + this->num_spatial_axes_)
      << "bottom num_axes may not change.";
  this->num_ = bottom[0]->count(0, this->channel_axis_);
  CHECK_EQ(bottom[0]->shape(this->channel_axis_), this->channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  // Setup the kernel share region dimensions.
  const int* kernel_share_data = kernel_share_.cpu_data();
  kernel_share_region_h_ = this->output_shape_[0] / kernel_share_data[0];
  kernel_share_region_w_ = this->output_shape_[1] / kernel_share_data[1];

  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    this->set_conv_out_spatial_dim(bottom[0]->count(first_spatial_axis));
  } else {
    this->set_conv_out_spatial_dim(top[0]->count(first_spatial_axis));
  }
  int conv_out_spatial_dim_ = this->get_conv_out_spatial_dim();
  int kernel_dim_ = this->get_kernel_dim();
  this->set_col_offset(kernel_dim_ * conv_out_spatial_dim_);
  int conv_out_channels = this->get_conv_out_channels();
  this->set_output_offset(conv_out_channels * conv_out_spatial_dim_ / this->group_);
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  this->conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = this->conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < this->num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(this->channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(this->channel_axis_ + i);
    }
  }
  // Setup the temporary blob which is used to store the convolutional result.
  vector<int> blob_buffer_shape(1, 1);
  blob_buffer_shape.push_back(this->num_output_ * kernel_share_data[0] * kernel_share_data[1]);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    blob_buffer_shape.push_back(this->output_shape_[i]);
  }
  blob_buffer_.Reshape(blob_buffer_shape);
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  Blob<Dtype>* col_buffer = this->get_col_buffer();
  this->col_buffer_shape_.clear();
  this->col_buffer_shape_.push_back(kernel_dim_ * this->group_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      this->col_buffer_shape_.push_back(this->input_shape(i + 1));
    } else {
      this->col_buffer_shape_.push_back(this->output_shape_[i]);
    }
  }
  col_buffer->Reshape(this->col_buffer_shape_);
  this->bottom_dim_ = bottom[0]->count(this->channel_axis_);
  this->top_dim_ = top[0]->count(this->channel_axis_);
  int conv_in_channels_  = this->get_conv_in_channels();
  conv_out_spatial_dim_ = this->get_conv_out_spatial_dim();
  this->set_num_kernels_im2col(conv_in_channels_ * conv_out_spatial_dim_);
  this->set_num_kernels_col2im(reverse_dimensions() ? this->top_dim_ : this->bottom_dim_);
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  this->out_spatial_dim_ = top[0]->count(first_spatial_axis);
  Blob<Dtype>* bias_multiplier = this->get_bias_multiplier();
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1, this->out_spatial_dim_);
    bias_multiplier->Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier->count(), Dtype(1),
        bias_multiplier->mutable_cpu_data());
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    Dtype* blob_buffer = blob_buffer_.mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          blob_buffer);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(blob_buffer, bias);
      }
      // Fuse the dimensions(1 * (N * map_fuse_num) * H * W) of convolutional output to
      // the dimensions(1 * N * H * W) of the layer output
      const int* kernel_share_data = kernel_share_.cpu_data();
      // The number of initial feature maps which fuse to one fused feature map
      int map_fuse_num = kernel_share_data[0] * kernel_share_data[1];
      // The number of initial maps
      int map_num = this->num_output_ * map_fuse_num;
      for (int k = 0; k < map_num; k++) {
        // The number of fused feature map
        int group = k / map_fuse_num;
        // The index of initial feature maps which fuse to one fused feature map
        int index = k % map_fuse_num;
        int out_spatial_dim = this->out_spatial_dim_;
        int out_spatial_dim_width = this->output_shape_[1];

        int height_offset = (index / kernel_share_data[1]) * out_spatial_dim_width;

        int map_offset = out_spatial_dim * map_fuse_num;
        for (int h = 0; h < kernel_share_region_h_; h++) {
          caffe_copy(kernel_share_region_w_, blob_buffer + group * map_offset + height_offset * kernel_share_region_h_ + index * (out_spatial_dim + kernel_share_region_w_)+ h * out_spatial_dim_width, 
            top_data + group * out_spatial_dim + height_offset * kernel_share_region_h_ + index * kernel_share_region_w_ + h * out_spatial_dim_width);
        }
      }
    }
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LocalConvolutionLayer);
#endif

INSTANTIATE_CLASS(LocalConvolutionLayer);

}  // namespace caffe
