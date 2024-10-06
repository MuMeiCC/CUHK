//
// OpenMP implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>    // OpenMP header
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_cores\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
	int num_threads = std::stoi(argv[3]);

	omp_set_num_threads(num_threads);

    // Separate R, G, B channels into three continuous arrays
    auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Transforming the R, G, B channels to Gray in parallel
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for default(none) shared(rChannel, gChannel, bChannel, filteredImage, input_jpeg,filter)
    for (int h = 1; h < input_jpeg.height-1; h++) {
		for (int w = 1; w < input_jpeg.width-1; w++) {
            int idx = h*input_jpeg.width+w;
			int sum_r=rChannel[idx-input_jpeg.width-1]*filter[0][0]+rChannel[idx-input_jpeg.width]*filter[0][1]+rChannel[idx-input_jpeg.width+1]*filter[0][2]
                    +rChannel[idx-1]*filter[1][0]+rChannel[idx]*filter[1][1]+rChannel[idx+1]*filter[1][2]
                    +rChannel[idx+input_jpeg.width-1]*filter[2][0]+rChannel[idx+input_jpeg.width]*filter[2][1]+rChannel[idx+input_jpeg.width+1]*filter[2][2];
			int sum_g=gChannel[idx-input_jpeg.width-1]*filter[0][0]+gChannel[idx-input_jpeg.width]*filter[0][1]+gChannel[idx-input_jpeg.width+1]*filter[0][2]
                    +gChannel[idx-1]*filter[1][0]+gChannel[idx]*filter[1][1]+gChannel[idx+1]*filter[1][2]
                    +gChannel[idx+input_jpeg.width-1]*filter[2][0]+gChannel[idx+input_jpeg.width]*filter[2][1]+gChannel[idx+input_jpeg.width+1]*filter[2][2];
            int sum_b=bChannel[idx-input_jpeg.width-1]*filter[0][0]+bChannel[idx-input_jpeg.width]*filter[0][1]+bChannel[idx-input_jpeg.width+1]*filter[0][2]
                    +bChannel[idx-1]*filter[1][0]+bChannel[idx]*filter[1][1]+bChannel[idx+1]*filter[1][2]
                    +bChannel[idx+input_jpeg.width-1]*filter[2][0]+bChannel[idx+input_jpeg.width]*filter[2][1]+bChannel[idx+input_jpeg.width+1]*filter[2][2];
			filteredImage[idx*input_jpeg.num_channels] = static_cast<unsigned char>(sum_r);
			filteredImage[idx*input_jpeg.num_channels+1] = static_cast<unsigned char>(sum_g);
			filteredImage[idx*input_jpeg.num_channels+2] = static_cast<unsigned char>(sum_b);
		}
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG GrayScale image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }

    // Release the allocated memory
    delete[] input_jpeg.buffer;
    delete[] rChannel;
    delete[] gChannel;
    delete[] bChannel;
    delete[] filteredImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
