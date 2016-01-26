#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/local_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void local_weight_gpu_kernel(const Dtype* top_diff, const Dtype* bottom,
  Dtype* weights_diff, const int local_weights_h_, const int local_weights_w_,
  const int num_output_) {
  int n = local_weights_h_ * local_weights_w_ * num_output_;
  CUDA_KERNEL_LOOP(i, n) {
    int col = i % local_weights_w_;
    int row = (i / local_weights_w_) % local_weights_h_;
    int k = (i / local_weights_w_) / local_weights_h_;
    weights_diff[i] += top_diff[k * local_weights_w_ + col] * 
      bottom[row * local_weights_w_ + col];
  }
}

template <typename Dtype>
void local_weight_gpu(const Dtype* top_diff, const Dtype* bottom,
  Dtype* weights_diff, const int local_weights_h_, const int local_weights_w_,
  const int num_output_) {
  local_weight_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(local_weights_h_ * local_weights_w_ * 
    num_output_), CAFFE_CUDA_NUM_THREADS>>>(top_diff, bottom, weights_diff, local_weights_h_, local_weights_w_, num_output_);
}

template <typename Dtype>
__global__ void local_backward_gpu_kernel(const Dtype* top_diff, const Dtype* weights,
  Dtype* bottom_diff, const int local_weights_dims_, const int local_weights_w_,
  const int num_output_) {
  int n = local_weights_dims_;
  CUDA_KERNEL_LOOP(i, n) {
    int col = i % local_weights_w_;
    for (int k = 0; k < num_output_; ++k) {
      bottom_diff[i] += top_diff[k * local_weights_w_ + col] * 
        weights[k * local_weights_dims_ + i];
    }
  }
}

template <typename Dtype>
void local_backward_gpu(const Dtype* top_diff, const Dtype* weights,
  Dtype* bottom_diff, const int local_weights_dims_, const int local_weights_w_,
  const int num_output_) {
  local_backward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(local_weights_dims_), 
    CAFFE_CUDA_NUM_THREADS>>>(top_diff, weights, bottom_diff, local_weights_dims_,
    local_weights_w_, num_output_);
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < num_; ++n) {
      forward_gpu_gemm(bottom_data + n * bottom_dim_, weight,
          top_data + n * top_dim_);
      if (bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        forward_gpu_bias(top_data + n * top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        backward_gpu_bias(bias_diff, top_diff + n * top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          weight_gpu_gemm(bottom_data + n * bottom_dim_,
              top_diff + n * top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          backward_gpu_gemm(top_diff + n * top_dim_, weight,
              bottom_diff + n * bottom_dim_);
        }
      }
    }
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int m = 0; m < num_output_; m++) {
    caffe_gpu_mul<Dtype>(local_weights_dims_, col_buff, weights + this->blobs_[0]->offset(m)
        ,output_buffer_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, local_weights_w_,
        local_weights_h_, (Dtype)1., weights_multiplier_.gpu_data(), 
        output_buffer_.gpu_data(), (Dtype)0., output + m * output_offset_);
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_add<Dtype>(num_output_ * out_spatial_dim_, bias, output, output);
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  caffe_gpu_set(col_buffer_.count(), (Dtype)0., col_buff);
  if (is_1x1_) {
    col_buff = input;
  }
  local_backward_gpu(output, weights, col_buff, local_weights_dims_, local_weights_w_, 
    num_output_);
  //for (int m = 0; m < num_output_; m++) {
  //  for (int k = 0; k < local_weights_h_; k++) {
  //    caffe_gpu_mul<Dtype>(local_weights_w_, output + m * output_offset_,
  //        weights + this->blobs_[0]->offset(m, 0, k), conv_output_buffer_.mutable_gpu_data());
  //    caffe_gpu_axpby<Dtype>(local_weights_w_, (Dtype)1., conv_output_buffer_.gpu_data(), 
  //        (Dtype)1., col_buff + k * output_offset_);
  //  }
  //}
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  local_weight_gpu(output, col_buff, weights, local_weights_h_, local_weights_w_, 
    num_output_);
  // for (int m = 0; m < num_output_; m++) {
  //   Dtype* local_weight_diff = weights + this->blobs_[0]->offset(m);
  //   for (int k = 0; k < local_weights_h_; k++) {
  //     caffe_gpu_mul<Dtype>(local_weights_w_, output + m * output_offset_,
  //         col_buff + k * output_offset_, output_buffer_.mutable_gpu_data() + k * 
  //         output_offset_);
  //   }
  //   caffe_gpu_axpby<Dtype>(local_weights_dims_, (Dtype)1., output_buffer_.gpu_data(), 
  //       (Dtype)1., local_weight_diff);
  // }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_add<Dtype>(num_output_ * out_spatial_dim_, bias, input, bias);
}

INSTANTIATE_LAYER_GPU_FUNCS(LocalConvolutionLayer);

}  // namespace caffe
