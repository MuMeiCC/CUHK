//
// OpenACC implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>

#include "utils.hpp"
// #include <openacc.h> // OpenACC Header

const double filter0 = 1.0/9,
            filter1 = 1.0/9,
            filter2 = 1.0/9,
            filter3 = 1.0/9,
            filter4 = 1.0/9,
            filter5 = 1.0/9,
            filter6 = 1.0/9,
            filter7 = 1.0/9,
            filter8 = 1.0/9;

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char *input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JPEGMeta input_jpeg = read_from_jpeg(input_filepath);
    // Computation: RGB to Gray
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    unsigned char *filteredImage = new unsigned char[width * height * num_channels];
    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }
#pragma acc enter data copyin(filteredImage[0 : width * height * num_channels], \
                              buffer[0 : width * height * num_channels])

#pragma acc update device(filteredImage[0 : width * height * num_channels], \
                          buffer[0 : width * height * num_channels])

    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel present(filteredImage[0 : width * height * num_channels], \
                             buffer[0 : width * height * num_channels]) \
    num_gangs(1024)
    {
#pragma acc loop independent
        for (int h = 1; h < height-1; h++) {
            for (int w = 1; w < width-1;w++) {
                int idx = h*width+w;
	            int sum_r=buffer[(idx-width-1)*num_channels]*filter0
                        +buffer[(idx-width)*num_channels]*filter1
                        +buffer[(idx-width+1)*num_channels]*filter2
                        +buffer[(idx-1)*num_channels]*filter3
                        +buffer[idx*num_channels]*filter4
                        +buffer[(idx+1)*num_channels]*filter5
                        +buffer[(idx+width-1)*num_channels]*filter6
                        +buffer[(idx+width)*num_channels]*filter7
                        +buffer[(idx+width+1)*num_channels]*filter8;

                int sum_g=buffer[(idx-width-1)*num_channels+1]*filter0
                        +buffer[(idx-width)*num_channels+1]*filter1
                        +buffer[(idx-width+1)*num_channels+1]*filter2
                        +buffer[(idx-1)*num_channels+1]*filter3
                        +buffer[idx*num_channels+1]*filter4
                        +buffer[(idx+1)*num_channels+1]*filter5
                        +buffer[(idx+width-1)*num_channels+1]*filter6
                        +buffer[(idx+width)*num_channels+1]*filter7
                        +buffer[(idx+width+1)*num_channels+1]*filter8;

                int sum_b=buffer[(idx-width-1)*num_channels+2]*filter0
                        +buffer[(idx-width)*num_channels+2]*filter1
                        +buffer[(idx-width+1)*num_channels+2]*filter2
                        +buffer[(idx-1)*num_channels+2]*filter3
                        +buffer[idx*num_channels+2]*filter4
                        +buffer[(idx+1)*num_channels+2]*filter5
                        +buffer[(idx+width-1)*num_channels+2]*filter6
                        +buffer[(idx+width)*num_channels+2]*filter7
                        +buffer[(idx+width+1)*num_channels+2]*filter8;

	            filteredImage[idx*num_channels] = static_cast<unsigned char>(sum_r);
	            filteredImage[idx*num_channels+1] = static_cast<unsigned char>(sum_g);
	            filteredImage[idx*num_channels+2] = static_cast<unsigned char>(sum_b);
	        }
		}
    }
    auto end_time = std::chrono::high_resolution_clock::now();
#pragma acc update self(filteredImage[0 : width * height * num_channels], \
                        buffer[0 : width * height * num_channels])

#pragma acc exit data copyout(filteredImage[0 : width * height * num_channels])

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Write Filtered Image to output JPEG
    const char *output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, width, height, num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
