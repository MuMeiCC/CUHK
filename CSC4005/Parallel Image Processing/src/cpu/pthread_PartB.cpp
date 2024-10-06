//
// Pthread implementation of image filtering
//

#include <iostream>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include "utils.hpp"

// Structure to pass data to each thread
struct ThreadData {
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int* offset;
    int width;
    int start;
    int end;
};

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

// Function to convert RGB to Grayscale for a portion of the image
void* filtering(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);

    for (int h = data->start; h < data->end; h++) {
        for (int w = 1; w < data->width-1; w++) {
            int idx = h*data->width+w;
            int sum_r=data->input_buffer[(idx-data->width-1)*3]*filter0
                    +data->input_buffer[(idx-data->width)*3]*filter1
                    +data->input_buffer[(idx-data->width+1)*3]*filter2
                    +data->input_buffer[(idx-1)*3]*filter3
                    +data->input_buffer[idx*3]*filter4
                    +data->input_buffer[(idx+1)*3]*filter5
                    +data->input_buffer[(idx+data->width-1)*3]*filter6
                    +data->input_buffer[(idx+data->width)*3]*filter7
                    +data->input_buffer[(idx+data->width+1)*3]*filter8;

            int sum_g=data->input_buffer[(idx-data->width-1)*3+1]*filter0
                    +data->input_buffer[(idx-data->width)*3+1]*filter1
                    +data->input_buffer[(idx-data->width+1)*3+1]*filter2
                    +data->input_buffer[(idx-1)*3+1]*filter3
                    +data->input_buffer[idx*3+1]*filter4
                    +data->input_buffer[(idx+1)*3+1]*filter5
                    +data->input_buffer[(idx+data->width-1)*3+1]*filter6
                    +data->input_buffer[(idx+data->width)*3+1]*filter7
                    +data->input_buffer[(idx+data->width+1)*3+1]*filter8;

            int sum_b=data->input_buffer[(idx-data->width-1)*3+2]*filter0
                    +data->input_buffer[(idx-data->width)*3+2]*filter1
                    +data->input_buffer[(idx-data->width+1)*3+2]*filter2
                    +data->input_buffer[(idx-1)*3+2]*filter3
                    +data->input_buffer[idx*3+2]*filter4
                    +data->input_buffer[(idx+1)*3+2]*filter5
                    +data->input_buffer[(idx+data->width-1)*3+2]*filter6
                    +data->input_buffer[(idx+data->width)*3+2]*filter7
                    +data->input_buffer[(idx+data->width+1)*3+2]*filter8;
            data->output_buffer[idx*3] = static_cast<unsigned char>(std::round(sum_r));
            data->output_buffer[idx*3+1] = static_cast<unsigned char>(std::round(sum_g));
            data->output_buffer[idx*3+2] = static_cast<unsigned char>(std::round(sum_b));
        }
    }

    return nullptr;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }

    int num_threads = std::stoi(argv[3]); // User-specified thread count

    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);

    // Computation: Filtering
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height*input_jpeg.num_channels];

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    auto start_time = std::chrono::high_resolution_clock::now();

    int chunk_size = (input_jpeg.height-2) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input_buffer = input_jpeg.buffer;
        thread_data[i].output_buffer = filteredImage;
        thread_data[i].width = input_jpeg.width;
        thread_data[i].start = i * chunk_size+1;
        thread_data[i].end = (i == num_threads - 1) ? (input_jpeg.height-1) : (i + 1) * chunk_size+1;

        pthread_create(&threads[i], nullptr, filtering, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Write filtered image to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
