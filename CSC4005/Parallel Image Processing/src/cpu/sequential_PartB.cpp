//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include <iostream>
#include <cmath>
#include <chrono>

#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter0 = 1.0/9,
            filter1 = 1.0/9,
            filter2 = 1.0/9,
            filter3 = 1.0/9,
            filter4 = 1.0/9,
            filter5 = 1.0/9,
            filter6 = 1.0/9,
            filter7 = 1.0/9,
            filter8 = 1.0/9;

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    // Nested for loop, please optimize it
    for (int height = 1; height < input_jpeg.height - 1; height++)
    {
        for (int width = 1; width < input_jpeg.width - 1; width++)
        {
            int idx = height*input_jpeg.width +width;
            int sum_r=input_jpeg.buffer[(idx-input_jpeg.width-1)*input_jpeg.num_channels]*filter0
                    +input_jpeg.buffer[(idx-input_jpeg.width)*input_jpeg.num_channels]*filter1
                    +input_jpeg.buffer[(idx-input_jpeg.width+1)*input_jpeg.num_channels]*filter2
                    +input_jpeg.buffer[(idx-1)*input_jpeg.num_channels]*filter3
                    +input_jpeg.buffer[idx*input_jpeg.num_channels]*filter4
                    +input_jpeg.buffer[(idx+1)*input_jpeg.num_channels]*filter5
                    +input_jpeg.buffer[(idx+input_jpeg.width-1)*input_jpeg.num_channels]*filter6
                    +input_jpeg.buffer[(idx+input_jpeg.width)*input_jpeg.num_channels]*filter7
                    +input_jpeg.buffer[(idx+input_jpeg.width+1)*input_jpeg.num_channels]*filter8;

            int sum_g=input_jpeg.buffer[(idx-input_jpeg.width-1)*input_jpeg.num_channels+1]*filter0
                    +input_jpeg.buffer[(idx-input_jpeg.width)*input_jpeg.num_channels+1]*filter1
                    +input_jpeg.buffer[(idx-input_jpeg.width+1)*input_jpeg.num_channels+1]*filter2
                    +input_jpeg.buffer[(idx-1)*input_jpeg.num_channels+1]*filter3
                    +input_jpeg.buffer[idx*input_jpeg.num_channels+1]*filter4
                    +input_jpeg.buffer[(idx+1)*input_jpeg.num_channels+1]*filter5
                    +input_jpeg.buffer[(idx+input_jpeg.width-1)*input_jpeg.num_channels+1]*filter6
                    +input_jpeg.buffer[(idx+input_jpeg.width)*input_jpeg.num_channels+1]*filter7
                    +input_jpeg.buffer[(idx+input_jpeg.width+1)*input_jpeg.num_channels+1]*filter8;

            int sum_b=input_jpeg.buffer[(idx-input_jpeg.width-1)*input_jpeg.num_channels+2]*filter0
                    +input_jpeg.buffer[(idx-input_jpeg.width)*input_jpeg.num_channels+2]*filter1
                    +input_jpeg.buffer[(idx-input_jpeg.width+1)*input_jpeg.num_channels+2]*filter2
                    +input_jpeg.buffer[(idx-1)*input_jpeg.num_channels+2]*filter3
                    +input_jpeg.buffer[idx*input_jpeg.num_channels+2]*filter4
                    +input_jpeg.buffer[(idx+1)*input_jpeg.num_channels+2]*filter5
                    +input_jpeg.buffer[(idx+input_jpeg.width-1)*input_jpeg.num_channels+2]*filter6
                    +input_jpeg.buffer[(idx+input_jpeg.width)*input_jpeg.num_channels+2]*filter7
                    +input_jpeg.buffer[(idx+input_jpeg.width+1)*input_jpeg.num_channels+2]*filter8;
                    
            filteredImage[idx * input_jpeg.num_channels]
                = static_cast<unsigned char>(std::round(sum_r));
            filteredImage[idx * input_jpeg.num_channels + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[idx * input_jpeg.num_channels + 2]
                = static_cast<unsigned char>(std::round(sum_b));
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
