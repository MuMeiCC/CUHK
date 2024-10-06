//
// MPI implementation of image filtering
//

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

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

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    // Read JPEG File
    const char * input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Divide the task using height of the picture
    int height_per_task = (input_jpeg.height-2) / numtasks;
    int left_height = (input_jpeg.height-2) % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_height = 0;

    cuts[0] = 1;
    for (int i = 0; i < numtasks; i++) {
        if (divided_left_height < left_height) {
            cuts[i+1] = cuts[i] + height_per_task + 1;
            divided_left_height++;
        } else cuts[i+1] = cuts[i] + height_per_task;
    }

    // The tasks for the master executor
    // 1. Transform the first division of the RGB contents to the Gray contents
    // 2. Receive the transformed Gray contents from slave executors
    // 3. Write the Gray contents to the JPEG File
    if (taskid == MASTER) {
        // Transform the first division of RGB Contents to the gray contents
        auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
        for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
            filteredImage[i] = 0;
        for (int h = cuts[MASTER]; h < cuts[MASTER+1]; h++)
        {
            for (int w = 1; w < input_jpeg.width - 1; w++)
            {
                int idx = h*input_jpeg.width+w;
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

        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filteredImage + cuts[i]*input_jpeg.width;
            int height = cuts[i+1] - cuts[i];
            MPI_Recv(start_pos, height, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);


        // Save the Gray Image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        // Release the memory
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }
    // The tasks for the slave executor
    // 1. Transform the RGB contents to the Gray contents
    // 2. Send the transformed Gray contents back to the master executor
    else {
        // Transform the RGB Contents to the gray contents
        int height = cuts[taskid + 1] - cuts[taskid];
        auto filteredImage = new unsigned char[input_jpeg.width * height * input_jpeg.num_channels];
        for (int i = 0; i < input_jpeg.width * height * input_jpeg.num_channels; ++i)
            filteredImage[i] = 0;
        for (int h = cuts[taskid]; h < cuts[taskid+1]; h++)
        {
            for (int w = 1; w < input_jpeg.width - 1; w++)
            {
                int idx = h*input_jpeg.width+w;
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

                int jdx = idx - cuts[taskid]*input_jpeg.width;
                filteredImage[jdx * input_jpeg.num_channels]
                    = static_cast<unsigned char>(std::round(sum_r));
                filteredImage[jdx * input_jpeg.num_channels + 1]
                    = static_cast<unsigned char>(std::round(sum_g));
                filteredImage[jdx * input_jpeg.num_channels + 2]
                    = static_cast<unsigned char>(std::round(sum_b));
            }
        }


        // Send the gray image back to the master
        MPI_Send(filteredImage, height, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        // Release the memory
        delete[] filteredImage;
    }

    MPI_Finalize();
    return 0;
}
