//
// CUDA implementation of image filtering
//

#include <iostream>

#include <cuda_runtime.h> // CUDA Header

#include "utils.hpp"

__constant__ double c_filter[3][3] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};


// CUDA kernel functon：Filtering
__global__ void filtering(const unsigned char* input, unsigned char* output,
                          int width, int height, int num_channels)
{
    extern __shared__ double filter[3][3];
    // __shared__ unsigned char tile[18][18][3];   // 18*18

    // Load filter coefficients into shared memory
    if (threadIdx.x < 9) {
        filter[threadIdx.x/3][threadIdx.x%3] = c_filter[threadIdx.x/3][threadIdx.x%3];
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx/width, x = idx%width;
    if(x>0&&x<width-1&&y>0&&y<height-1) {
        int sum_r=input[(idx-width-1)*num_channels]*filter[0][0]
                +input[(idx-width)*num_channels]*filter[0][1]
                +input[(idx-width+1)*num_channels]*filter[0][2]
                +input[(idx-1)*num_channels]*filter[1][0]
                +input[idx*num_channels]*filter[1][1]
                +input[(idx+1)*num_channels]*filter[1][2]
                +input[(idx+width-1)*num_channels]*filter[2][0]
                +input[(idx+width)*num_channels]*filter[2][1]
                +input[(idx+width+1)*num_channels]*filter[2][2];

        int sum_g=input[(idx-width-1)*num_channels+1]*filter[0][0]
                +input[(idx-width)*num_channels+1]*filter[0][1]
                +input[(idx-width+1)*num_channels+1]*filter[0][2]
                +input[(idx-1)*num_channels+1]*filter[1][0]
                +input[idx*num_channels+1]*filter[1][1]
                +input[(idx+1)*num_channels+1]*filter[1][2]
                +input[(idx+width-1)*num_channels+1]*filter[2][0]
                +input[(idx+width)*num_channels+1]*filter[2][1]
                +input[(idx+width+1)*num_channels+1]*filter[2][2];

        int sum_b=input[(idx-width-1)*num_channels+2]*filter[0][0]
                +input[(idx-width)*num_channels+2]*filter[0][1]
                +input[(idx-width+1)*num_channels+2]*filter[0][2]
                +input[(idx-1)*num_channels+2]*filter[1][0]
                +input[idx*num_channels+2]*filter[1][1]
                +input[(idx+1)*num_channels+2]*filter[1][2]
                +input[(idx+width-1)*num_channels+2]*filter[2][0]
                +input[(idx+width)*num_channels+2]*filter[2][1]
                +input[(idx+width+1)*num_channels+2]*filter[2][2];

        output[idx*num_channels] = static_cast<unsigned char>(sum_r);
        output[idx*num_channels+1] = static_cast<unsigned char>(sum_g);
        output[idx*num_channels+2] = static_cast<unsigned char>(sum_b);
    }

}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width*input_jpeg.height*input_jpeg.num_channels];   // num_channels = 1
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input,
                input_jpeg.width*input_jpeg.height*input_jpeg.num_channels*sizeof(unsigned char));
    cudaMalloc((void**)&d_output,
               input_jpeg.width*input_jpeg.height*input_jpeg.num_channels*sizeof(unsigned char));
    // Copy input data from host to device
    cudaMemcpy(d_input, input_jpeg.buffer,
               input_jpeg.width*input_jpeg.height*input_jpeg.num_channels *
                   sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    // Computation: Filtering
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // dim3 blockSize(16, 16);
    // dim3 blockNum((input_jpeg.width + blockSize.x - 1) / blockSize.x, (input_jpeg.height + blockSize.y - 1) / blockSize.y);
    int blockSize = 512;
    int blockNum = (input_jpeg.width * input_jpeg.height) / blockSize + 1;
    cudaEventRecord(start, 0); // GPU start time
    // Kernel function
    filtering<<<blockNum, blockSize, 18*18*3*sizeof(unsigned char) + 3*3*sizeof(double)>>>(
        d_input, d_output, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    // Write Filtered Image to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
