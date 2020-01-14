//
// Created by mingkon on 1/10/20.
//

#include "acce.cuh"

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}


__global__ void foo()
{
    printf("CUDA!\n");

}

__global__ void kernelGetRGBs()
{

    printf("kernelGetRGBs!\n");
}

__global__ void kernelGetAver()
{
    printf("kernelGetAver!\n");
}

__global__ void kernelSetWB(
        uchar3 * const bgr_d,
        uchar3 * wb_bgr_d,
        float blueAver, float greenAver, float redAver,
        int _image_height, int _image_width)
{
    printf("kernelSetWB in GPU !\n");

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < _image_width && idy < _image_height)
    {
        uchar3 rgb = bgr_d[idy * _image_width + idx];
        float blue = static_cast<float>(rgb.x / blueAver * 255);
        float green = static_cast<float>(rgb.y / greenAver * 255);
        float red = static_cast<float>(rgb.z / redAver * 255);

        if (blue > 255) {
            blue = 255;
        } else if (blue < 0) {
            blue = 0;
        }
        if (green > 255) {
            green = 255;
        } else if (green < 0) {
            green = 0;
        }
        if (red > 255) {
            red = 255;
        } else if (red < 0) {
            red = 0;
        }

        wb_bgr_d[idy * _image_width + idx].x = static_cast<uchar>(blue);
        wb_bgr_d[idy * _image_width + idx].y = static_cast<uchar>(green);
        wb_bgr_d[idy * _image_width + idx].z = static_cast<uchar>(red);

    }



}

void useCUDA()
{

    foo<<<1,5>>>();
    CHECK(cudaDeviceSynchronize());

}


void getRGBs(std::vector<float> * res,cv::Mat * Image)
{
    kernelGetRGBs<<<1,5>>>();
    CHECK(cudaDeviceSynchronize());

}


void getAver()
{
    kernelGetAver<<<1,5>>>();
    CHECK(cudaDeviceSynchronize());

}


void setWB(const cv::Mat & img, cv::Mat & processedImg,
           float blueAver, float greenAver, float redAver, int _image_height, int _image_width)
{

    uchar3 * bgr_d, * wb_bgr_d;

    cudaMalloc((void**)&bgr_d, _image_height*_image_width*sizeof(uchar3));
    cudaMalloc((void**)&wb_bgr_d, _image_height*_image_width*sizeof(uchar3));

    cudaMemcpy(bgr_d, img.data, _image_height*_image_width*sizeof(uchar3), cudaMemcpyHostToDevice);


    dim3 dimBlock(32, 32);
    dim3 dimGrid((_image_width + dimBlock.x - 1) / dimBlock.x,
                       (_image_height + dimBlock.y - 1) / dimBlock.y);

    kernelSetWB<<<dimGrid,dimBlock>>>(bgr_d, wb_bgr_d, blueAver, greenAver, redAver,_image_height, _image_width);
    CHECK(cudaDeviceSynchronize());

}