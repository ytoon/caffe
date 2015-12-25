#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/local_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  LocalConvolutionParameter local_conv_param = this->layer_param_.local_convolution_param();
  force_nd_im2col_ = local_conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(local_conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  LOG(INFO) << "first_spatial_axis " << first_spatial_axis;
  const int num_axes = bottom[0]->num_axes();
  LOG(INFO) << "num_axes " << num_axes;
  num_spatial_axes_ = num_axes - first_spatial_axis;
  LOG(INFO) << "num_spatial_axes_ " << num_spatial_axes_;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  LOG(INFO) << "spatial_dim_blob_shape " << spatial_dim_blob_shape[0];
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (local_conv_param.has_kernel_h() || local_conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, local_conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = local_conv_param.kernel_h();
    kernel_shape_data[1] = local_conv_param.kernel_w();
  } else {
    const int num_kernel_dims = local_conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            local_conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    LOG(INFO) << "kernel_shape_data " << kernel_shape_data[i];
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (local_conv_param.has_stride_h() || local_conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, local_conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = local_conv_param.stride_h();
    stride_data[1] = local_conv_param.stride_w();
  } else {
    const int num_stride_dims = local_conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          local_conv_param.stride((num_stride_dims == 1) ? 0 : i);
      LOG(INFO) << "stride_data " << stride_data[i];
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (local_conv_param.has_pad_h() || local_conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, local_conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = local_conv_param.pad_h();
    pad_data[1] = local_conv_param.pad_w();
  } else {
    const int num_pad_dims = local_conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          local_conv_param.pad((num_pad_dims == 1) ? 0 : i);
      LOG(INFO) << "pad_data " << pad_data[i];
    }
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  LOG(INFO) << "is_1x1_ " << is_1x1_;
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  LOG(INFO) << "channels_ " << channels_;
  num_output_ = this->layer_param_.local_convolution_param().num_output();
  LOG(INFO) << "num_output_ " << num_output_;
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.local_convolution_param().group();
  LOG(INFO) << "group_ " << group_;
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  bottom_shape_ = &bottom[0]->shape();
  LOG(INFO) << "bottom_shape_ " << (*bottom_shape_)[0];
  LOG(INFO) << "bottom_shape_ " << (*bottom_shape_)[1];
  LOG(INFO) << "bottom_shape_ " << (*bottom_shape_)[2];
  LOG(INFO) << "bottom_shape_ " << (*bottom_shape_)[3];
  compute_output_shape();
  LOG(INFO) << "output_shape_ " << output_shape_[0];
  LOG(INFO) << "output_shape_ " << output_shape_[1];
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the local filter weights
  // - blobs_[1] holds the local biases (optional)
  local_weights_h_ = channels_ * kernel_shape_data[0] * kernel_shape_data[1];
  local_weights_w_ = output_shape_[0] * output_shape_[1];
  local_weights_dims_ = local_weights_h_ * local_weights_w_;
  LOG(INFO) << "local_weights_h_ " << local_weights_h_;
  LOG(INFO) << "local_weights_w_ " << local_weights_w_;
  LOG(INFO) << "local_weights_dims_ " << local_weights_dims_;
  // TODO: local conv with group_
  vector<int> weight_shape(2, 1);
  weight_shape[0] = conv_out_channels_;
  weight_shape.push_back(local_weights_h_);
  weight_shape.push_back(local_weights_w_);
  LOG(INFO) << "weight_shape " << weight_shape[0];
  LOG(INFO) << "weight_shape " << weight_shape[1];
  LOG(INFO) << "weight_shape " << weight_shape[2];
  LOG(INFO) << "weight_shape " << weight_shape[3];
  bias_term_ = this->layer_param_.local_convolution_param().bias_term();
  LOG(INFO) << "bias_term_ " << bias_term_;
  vector<int> bias_shape(2, 1);
  bias_shape.push_back(conv_out_channels_);
  bias_shape.push_back(local_weights_w_);
  LOG(INFO) << "bias_shape " << bias_shape[0];
  LOG(INFO) << "bias_shape " << bias_shape[1];
  LOG(INFO) << "bias_shape " << bias_shape[2];
  LOG(INFO) << "bias_shape " << bias_shape[3];
  LOG(INFO) << "blob_size " << this->blobs_.size();
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
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
    const Dtype* weight = this->blobs_[0]->cpu_data();
    LOG(INFO) << "weight[0] " << weight[0];
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.local_convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // TODO: the kernel_dim_ and weight_offset_ should be thinked twice.
  // kernel_dim_ = this->blobs_[0]->count(1);
  // weight_offset_ = num_output_ * kernel_dim_;
  output_offset_ = output_shape_[0] * output_shape_[1];
  // LOG(INFO) << "kernel_dim_ " << kernel_dim_;
  // LOG(INFO) << "weight_offset_ " << weight_offset_;
  LOG(INFO) << "output_offset_ " << output_offset_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = kernel_shape_.cpu_data();
  const int* stride_data = stride_.cpu_data();
  const int* pad_data = pad_.cpu_data();
  output_shape_.clear();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    output_shape_.push_back(output_dim);

  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  // LOG(INFO) << "num_" << num_;
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  // bottom_shape_ = &bottom[0]->shape();
  // compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(local_weights_h_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  // LOG(INFO) << "col_buffer_shape_ " << col_buffer_shape_[0];
  // LOG(INFO) << "col_buffer_shape_ " << col_buffer_shape_[1];
  // LOG(INFO) << "col_buffer_shape_ " << col_buffer_shape_[2];
  col_buffer_.Reshape(col_buffer_shape_);
  // 
  vector<int> output_buffer_shape(2, 1);
  output_buffer_shape.push_back(local_weights_h_);
  output_buffer_shape.push_back(local_weights_w_);
  output_buffer_.Reshape(output_buffer_shape);
  // 
  vector<int> conv_output_buffer_shape(3, 1);
  conv_output_buffer_shape.push_back(local_weights_w_);
  conv_output_buffer_.Reshape(conv_output_buffer_shape);
  
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  // LOG(INFO) << "bottom_dim_ " << bottom_dim_;
  // LOG(INFO) << "top_dim_ " << top_dim_;
  // TODO: the num_kernels_im2col_ and num_kernels_col2im_ should be thinked twice.
  // num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  // num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  // LOG(INFO) << "out_spatial_dim_ " << out_spatial_dim_;
  vector<int> weights_multiplier_shape(3, 1);
  weights_multiplier_shape.push_back(local_weights_h_);
  weights_multiplier_.Reshape(weights_multiplier_shape);
  caffe_set(weights_multiplier_.count(), Dtype(1),
      weights_multiplier_.mutable_cpu_data());
  // LOG(INFO) << "weights_multiplier_ " << weights_multiplier_.count();
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // LOG(INFO) << "weight[0] " << weight[0];
  // LOG(INFO) << "bottom_size " << bottom.size();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < num_; ++n) {
      forward_cpu_gemm(bottom_data + n * bottom_dim_, weight,
          top_data + n * top_dim_);
      if (bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        forward_cpu_bias(top_data + n * top_dim_, bias);
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
    // LOG(INFO) << "top size " << top[i]->count();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // LOG(INFO) << "bottom_diff " << bottom_diff[100];
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < num_; ++n) {
        backward_cpu_bias(bias_diff, top_diff + n * top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          weight_cpu_gemm(bottom_data + n * bottom_dim_,
              top_diff + n * top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          backward_cpu_gemm(top_diff + n * top_dim_, weight,
              bottom_diff + n * bottom_dim_);
        }
      }
    }
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
    // LOG(INFO) << "input[0] " << input[0];
    // LOG(INFO) << "col_buff[0] " << col_buff[5];
  }
  for (int m = 0; m < num_output_; m++) {
    caffe_mul<Dtype>(local_weights_dims_, col_buff, weights + this->blobs_[0]->offset(m),
        output_buffer_.mutable_cpu_data());
    // const Dtype* output_buffer = output_buffer_.cpu_data();
    // LOG(INFO) << "output_buffer_[0] " << output_buffer[0];
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, local_weights_w_,
        local_weights_h_, (Dtype)1., weights_multiplier_.cpu_data(), 
        output_buffer_.cpu_data(), (Dtype)0., output + m * output_offset_);
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_add<Dtype>(num_output_ * out_spatial_dim_, bias, output, output);
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  caffe_set(col_buffer_.count(), (Dtype)0., col_buff);
  if (is_1x1_) {
    col_buff = input;
  }
  for (int m = 0; m < num_output_; m++) {
    for (int k = 0; k < local_weights_h_; k++) {
      caffe_mul<Dtype>(local_weights_w_, output + m * output_offset_,
          weights + this->blobs_[0]->offset(m, 0, k), conv_output_buffer_.mutable_cpu_data());
      caffe_cpu_axpby<Dtype>(local_weights_w_, (Dtype)1., conv_output_buffer_.cpu_data(), 
          (Dtype)1., col_buff + k * output_offset_);
    }
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int m = 0; m < num_output_; m++) {
    Dtype* local_weight_diff = weights + this->blobs_[0]->offset(m);
    for (int k = 0; k < local_weights_h_; k++) {
      caffe_mul<Dtype>(local_weights_w_, output + m * output_offset_,
          col_buff + k * output_offset_, output_buffer_.mutable_cpu_data() + k * output_offset_);
    }
    caffe_cpu_axpby<Dtype>(local_weights_dims_, (Dtype)1., output_buffer_.cpu_data(), 
        (Dtype)1., local_weight_diff);
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_add<Dtype>(num_output_ * out_spatial_dim_, bias, input, bias);
}

#ifdef CPU_ONLY
STUB_GPU(LocalConvolutionLayer);
#endif

INSTANTIATE_CLASS(LocalConvolutionLayer);
REGISTER_LAYER_CLASS(LocalConvolution);

}  // namespace caffe
