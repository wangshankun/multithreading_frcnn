#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/calcu_pthread.h"


typedef struct {
    float*  data_im;
    int     channels;
    int     height;
    int     width; 
    int     kernel_h;
    int     kernel_w;
    int     pad_h;
    int     pad_w;
    int     stride_h;
    int     stride_w;
    int     dilation_h;
    int     dilation_w;
    int     output_h;
    int     output_w;
    int     channel_size;
    int     data_col_size;
    float*  data_col;
    int*    range_channel;
} im2col_arg_t;

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

static void divide(int M, int* range_M)
{
    int dx = M%IM2COL_MAX_CPU_NUMBER;
    int dy = M/IM2COL_MAX_CPU_NUMBER;
    int index = 0;
    int i;
    for(i = 0;i < IM2COL_MAX_CPU_NUMBER + 1; i++)
    {
        range_M[i] = index;
        if(i < dx)
        {
            index = index + dy + 1;
        }
        else
        {
            index = index + dy;
        }
    }
}

void im2col_inner_thread(void* set_args, int pos)
{
      im2col_arg_t* args    =  (im2col_arg_t*)set_args;
      float*  data_im       = args -> data_im      ;
      //int     channels      = args -> channels     ;
      int     height        = args -> height       ;
      int     width         = args -> width        ; 
      int     kernel_h      = args -> kernel_h     ;
      int     kernel_w      = args -> kernel_w     ;
      int     pad_h         = args -> pad_h        ;
      int     pad_w         = args -> pad_w        ;
      int     stride_h      = args -> stride_h     ;
      int     stride_w      = args -> stride_w     ;
      int     dilation_h    = args -> dilation_h   ;
      int     dilation_w    = args -> dilation_w   ;
      int     output_h      = args -> output_h     ;
      int     output_w      = args -> output_w     ;
      int     channel_size  = args -> channel_size ;
      int     data_col_size = args -> data_col_size;
      float*  data_col      = args -> data_col     ;
      int*    range_channel = args -> range_channel;
      
      data_im  += range_channel[pos] * channel_size;
      data_col += range_channel[pos] * data_col_size;
      for(int channel = range_channel[pos]; channel < range_channel[pos + 1]; channel++, data_im += channel_size)
      {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = -pad_h + kernel_row * dilation_h;
            for (int output_rows = output_h; output_rows; output_rows--) {
              if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                for (int output_cols = output_w; output_cols; output_cols--) {
                  *(data_col++) = 0;
                }
              } else {
                int input_col = -pad_w + kernel_col * dilation_w;
                for (int output_col = output_w; output_col; output_col--) {
                  if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                    *(data_col++) = data_im[input_row * width + input_col];
                  } else {
                    *(data_col++) = 0;
                  }
                  input_col += stride_w;
                }
              }
              input_row += stride_h;
            }
          }
        }
    }
  
}

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.

double im2col_elapsed_time=0;
template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    const int output_h      = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w      = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size  = height * width;
    const int data_col_size = output_h * output_w * kernel_h * kernel_w;
  
    queue_t           queue_Q[IM2COL_MAX_CPU_NUMBER];
    int               range_channel[IM2COL_MAX_CPU_NUMBER + 1];
    im2col_arg_t      ins_args;

    ins_args.data_im       = (float*)data_im;
    //ins_args.channels      = channels;
    ins_args.height        = height;
    ins_args.width         = width; 
    ins_args.kernel_h      = kernel_h;
    ins_args.kernel_w      = kernel_w;
    ins_args.pad_h         = pad_h;
    ins_args.pad_w         = pad_w;
    ins_args.stride_h      = stride_h;
    ins_args.stride_w      = stride_w;
    ins_args.dilation_h    = dilation_h;
    ins_args.dilation_w    = dilation_w;
    ins_args.output_h      = output_h;
    ins_args.output_w      = output_w;
    ins_args.channel_size  = channel_size;
    ins_args.data_col_size = data_col_size;
    ins_args.data_col      = (float*)data_col;
    ins_args.range_channel = range_channel;
    
    int i;
    divide(channels, range_channel);
    for (i = 0; i < IM2COL_MAX_CPU_NUMBER; i++)
    {    
        queue_Q[i].routine     = im2col_inner_thread;
        queue_Q[i].position    = i;
        queue_Q[i].args        = &ins_args;
    }

    all_sub_pthread_exec(queue_Q, IM2COL_MAX_CPU_NUMBER);

  
    clock_gettime(CLOCK_MONOTONIC, &finish);
    im2col_elapsed_time += (finish.tv_sec - start.tv_sec);
    im2col_elapsed_time += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    //printf("im2col_elapsed_time:%f\r\n",im2col_elapsed_time);
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);

template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int num_spatial_axes, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_output) {
  if (!im2col) {
    int im_size = im_shape[0];
    for (int i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    caffe_set(im_size, Dtype(0), data_output);
  }
  int kernel_size = 1;
  for (int i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int channels_col = col_shape[0];
  vector<int> d_offset(num_spatial_axes, 0);
  vector<int> d_iter(num_spatial_axes, 0);
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c_col;
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int d = d_iter[d_i];
        const int d_im = d * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, dilation, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);


}  // namespace caffe
