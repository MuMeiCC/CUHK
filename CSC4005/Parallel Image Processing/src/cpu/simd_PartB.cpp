//
// SIMD (AVX2) implementation of transferring a JPEG picture from RGB to gray
//

#include <iostream>
#include <chrono>

#include <immintrin.h>

#include "utils.hpp"

// Set SIMD scalars
const __m256 f0 = _mm256_set1_ps(0.11111111f);
const __m256 f1 = _mm256_set1_ps(0.11111111f);
const __m256 f2 = _mm256_set1_ps(0.11111111f);
const __m256 f3 = _mm256_set1_ps(0.11111111f);
const __m256 f4 = _mm256_set1_ps(0.11111111f);
const __m256 f5 = _mm256_set1_ps(0.11111111f);
const __m256 f6 = _mm256_set1_ps(0.11111111f);
const __m256 f7 = _mm256_set1_ps(0.11111111f);
const __m256 f8 = _mm256_set1_ps(0.11111111f);

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Transform the RGB Contents to the gray contents
    auto filteredRed = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto filteredGreen = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto filteredBlue = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    // Prepross, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Mask used for shuffling when store int32s to u_int8 arrays
    // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
    __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12,
                                    -1, -1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1, -1, -1, -1);

    // Using SIMD to accelerate the transformation
    auto start_time = std::chrono::high_resolution_clock::now();    // Start recording time
    for (int h = 1; h < input_jpeg.height-1; h++) {
        for (int w = 1; w < input_jpeg.width-1; w+=8){
            int idx = h*input_jpeg.width+w;

            __m128i red_chars = _mm_loadu_si128((__m128i*)(reds+idx));
            __m256i red_ints = _mm256_cvtepu8_epi32(red_chars);
            __m256 red_floats = _mm256_cvtepi32_ps(red_ints);
            __m256 red_results = _mm256_mul_ps(red_floats, f4);

            red_chars = _mm_loadu_si128((__m128i*)(reds+idx-input_jpeg.width-1));
            red_ints = _mm256_cvtepu8_epi32(red_chars);
            red_floats = _mm256_cvtepi32_ps(red_ints);
            __m256 red_around = _mm256_mul_ps(red_floats, f0);
            red_results = _mm256_add_ps(red_results, red_around);
            
            red_chars = _mm_loadu_si128((__m128i*)(reds+idx-input_jpeg.width));
            red_ints = _mm256_cvtepu8_epi32(red_chars);
            red_floats = _mm256_cvtepi32_ps(red_ints);
            red_around = _mm256_mul_ps(red_floats, f1);
            red_results = _mm256_add_ps(red_results, red_around);
            
            red_chars = _mm_loadu_si128((__m128i*)(reds+idx-input_jpeg.width+1));
            red_ints = _mm256_cvtepu8_epi32(red_chars);
            red_floats = _mm256_cvtepi32_ps(red_ints);
            red_around = _mm256_mul_ps(red_floats, f2);
            red_results = _mm256_add_ps(red_results, red_around);
            
            red_chars = _mm_loadu_si128((__m128i*)(reds+idx-1));
            red_ints = _mm256_cvtepu8_epi32(red_chars);
            red_floats = _mm256_cvtepi32_ps(red_ints);
            red_around = _mm256_mul_ps(red_floats, f3);
            red_results = _mm256_add_ps(red_results, red_around);
            
            red_chars = _mm_loadu_si128((__m128i*)(reds+idx+1));
            red_ints = _mm256_cvtepu8_epi32(red_chars);
            red_floats = _mm256_cvtepi32_ps(red_ints);
            red_around = _mm256_mul_ps(red_floats, f5);
            red_results = _mm256_add_ps(red_results, red_around);
            
            red_chars = _mm_loadu_si128((__m128i*)(reds+idx+input_jpeg.width-1));
            red_ints = _mm256_cvtepu8_epi32(red_chars);
            red_floats = _mm256_cvtepi32_ps(red_ints);
            red_around = _mm256_mul_ps(red_floats, f6);
            red_results = _mm256_add_ps(red_results, red_around);
            
            red_chars = _mm_loadu_si128((__m128i*)(reds+idx+input_jpeg.width));
            red_ints = _mm256_cvtepu8_epi32(red_chars);
            red_floats = _mm256_cvtepi32_ps(red_ints);
            red_around = _mm256_mul_ps(red_floats, f7);
            red_results = _mm256_add_ps(red_results, red_around);
            
            red_chars = _mm_loadu_si128((__m128i*)(reds+idx+input_jpeg.width+1));
            red_ints = _mm256_cvtepu8_epi32(red_chars);
            red_floats = _mm256_cvtepi32_ps(red_ints);
            red_around = _mm256_mul_ps(red_floats, f8);
            red_results = _mm256_add_ps(red_results, red_around);
            
            __m256i red_results_ints = _mm256_cvtps_epi32(red_results);
            __m128i red_low = _mm256_castsi256_si128(red_results_ints);
            __m128i red_high = _mm256_extracti128_si256(red_results_ints, 1);
            __m128i tred_low = _mm_shuffle_epi8(red_low, shuffle);
            __m128i tred_high = _mm_shuffle_epi8(red_high, shuffle);
            _mm_storeu_si128((__m128i*)(&filteredRed[idx]), tred_low);
            _mm_storeu_si128((__m128i*)(&filteredRed[idx+4]), tred_high);
            
            __m128i green_chars = _mm_loadu_si128((__m128i*)(greens+idx));
            __m256i green_ints = _mm256_cvtepu8_epi32(green_chars);
            __m256 green_floats = _mm256_cvtepi32_ps(green_ints);
            __m256 green_results = _mm256_mul_ps(green_floats, f4);

            green_chars = _mm_loadu_si128((__m128i*)(greens+idx-input_jpeg.width-1));
            green_ints = _mm256_cvtepu8_epi32(green_chars);
            green_floats = _mm256_cvtepi32_ps(green_ints);
            __m256 green_around = _mm256_mul_ps(green_floats, f0);
            green_results = _mm256_add_ps(green_results, green_around);
            
            green_chars = _mm_loadu_si128((__m128i*)(greens+idx-input_jpeg.width));
            green_ints = _mm256_cvtepu8_epi32(green_chars);
            green_floats = _mm256_cvtepi32_ps(green_ints);
            green_around = _mm256_mul_ps(green_floats, f1);
            green_results = _mm256_add_ps(green_results, green_around);
            
            green_chars = _mm_loadu_si128((__m128i*)(greens+idx-input_jpeg.width+1));
            green_ints = _mm256_cvtepu8_epi32(green_chars);
            green_floats = _mm256_cvtepi32_ps(green_ints);
            green_around = _mm256_mul_ps(green_floats, f2);
            green_results = _mm256_add_ps(green_results, green_around);
            
            green_chars = _mm_loadu_si128((__m128i*)(greens+idx-1));
            green_ints = _mm256_cvtepu8_epi32(green_chars);
            green_floats = _mm256_cvtepi32_ps(green_ints);
            green_around = _mm256_mul_ps(green_floats, f3);
            green_results = _mm256_add_ps(green_results, green_around);
            
            green_chars = _mm_loadu_si128((__m128i*)(greens+idx+1));
            green_ints = _mm256_cvtepu8_epi32(green_chars);
            green_floats = _mm256_cvtepi32_ps(green_ints);
            green_around = _mm256_mul_ps(green_floats, f5);
            green_results = _mm256_add_ps(green_results, green_around);
            
            green_chars = _mm_loadu_si128((__m128i*)(greens+idx+input_jpeg.width-1));
            green_ints = _mm256_cvtepu8_epi32(green_chars);
            green_floats = _mm256_cvtepi32_ps(green_ints);
            green_around = _mm256_mul_ps(green_floats, f6);
            green_results = _mm256_add_ps(green_results, green_around);
            
            green_chars = _mm_loadu_si128((__m128i*)(greens+idx+input_jpeg.width));
            green_ints = _mm256_cvtepu8_epi32(green_chars);
            green_floats = _mm256_cvtepi32_ps(green_ints);
            green_around = _mm256_mul_ps(green_floats, f7);
            green_results = _mm256_add_ps(green_results, green_around);
            
            green_chars = _mm_loadu_si128((__m128i*)(greens+idx+input_jpeg.width+1));
            green_ints = _mm256_cvtepu8_epi32(green_chars);
            green_floats = _mm256_cvtepi32_ps(green_ints);
            green_around = _mm256_mul_ps(green_floats, f8);
            green_results = _mm256_add_ps(green_results, green_around);
            
            __m256i green_results_ints = _mm256_cvtps_epi32(green_results);
            __m128i green_low = _mm256_castsi256_si128(green_results_ints);
            __m128i green_high = _mm256_extracti128_si256(green_results_ints, 1);
            __m128i tgreen_low = _mm_shuffle_epi8(green_low, shuffle);
            __m128i tgreen_high = _mm_shuffle_epi8(green_high, shuffle);
            _mm_storeu_si128((__m128i*)(&filteredGreen[idx]), tgreen_low);
            _mm_storeu_si128((__m128i*)(&filteredGreen[idx+4]), tgreen_high);
            
            __m128i blue_chars = _mm_loadu_si128((__m128i*)(blues+idx));
            __m256i blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            __m256 blue_floats = _mm256_cvtepi32_ps(blue_ints);
            __m256 blue_results = _mm256_mul_ps(blue_floats, f4);

            blue_chars = _mm_loadu_si128((__m128i*)(blues+idx-input_jpeg.width-1));
            blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            blue_floats = _mm256_cvtepi32_ps(blue_ints);
            __m256 blue_around = _mm256_mul_ps(blue_floats, f0);
            blue_results = _mm256_add_ps(blue_results, blue_around);
            
            blue_chars = _mm_loadu_si128((__m128i*)(blues+idx-input_jpeg.width));
            blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            blue_floats = _mm256_cvtepi32_ps(blue_ints);
            blue_around = _mm256_mul_ps(blue_floats, f1);
            blue_results = _mm256_add_ps(blue_results, blue_around);
            
            blue_chars = _mm_loadu_si128((__m128i*)(blues+idx-input_jpeg.width+1));
            blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            blue_floats = _mm256_cvtepi32_ps(blue_ints);
            blue_around = _mm256_mul_ps(blue_floats, f2);
            blue_results = _mm256_add_ps(blue_results, blue_around);
            
            blue_chars = _mm_loadu_si128((__m128i*)(blues+idx-1));
            blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            blue_floats = _mm256_cvtepi32_ps(blue_ints);
            blue_around = _mm256_mul_ps(blue_floats, f3);
            blue_results = _mm256_add_ps(blue_results, blue_around);
            
            blue_chars = _mm_loadu_si128((__m128i*)(blues+idx+1));
            blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            blue_floats = _mm256_cvtepi32_ps(blue_ints);
            blue_around = _mm256_mul_ps(blue_floats, f5);
            blue_results = _mm256_add_ps(blue_results, blue_around);
            
            blue_chars = _mm_loadu_si128((__m128i*)(blues+idx+input_jpeg.width-1));
            blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            blue_floats = _mm256_cvtepi32_ps(blue_ints);
            blue_around = _mm256_mul_ps(blue_floats, f6);
            blue_results = _mm256_add_ps(blue_results, blue_around);
            
            blue_chars = _mm_loadu_si128((__m128i*)(blues+idx+input_jpeg.width));
            blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            blue_floats = _mm256_cvtepi32_ps(blue_ints);
            blue_around = _mm256_mul_ps(blue_floats, f7);
            blue_results = _mm256_add_ps(blue_results, blue_around);
            
            blue_chars = _mm_loadu_si128((__m128i*)(blues+idx+input_jpeg.width+1));
            blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            blue_floats = _mm256_cvtepi32_ps(blue_ints);
            blue_around = _mm256_mul_ps(blue_floats, f8);
            blue_results = _mm256_add_ps(blue_results, blue_around);
            
            __m256i blue_results_ints = _mm256_cvtps_epi32(blue_results);
            __m128i blue_low = _mm256_castsi256_si128(blue_results_ints);
            __m128i blue_high = _mm256_extracti128_si256(blue_results_ints, 1);
            __m128i tblue_low = _mm_shuffle_epi8(blue_low, shuffle);
            __m128i tblue_high = _mm_shuffle_epi8(blue_high, shuffle);
            _mm_storeu_si128((__m128i*)(&filteredBlue[idx]), tblue_low);
            _mm_storeu_si128((__m128i*)(&filteredBlue[idx+4]), tblue_high);
            
        }
    }
    for(int i=0;i<input_jpeg.width*input_jpeg.height;i++){
        filteredImage[i*input_jpeg.num_channels] = filteredRed[i];
        filteredImage[i*input_jpeg.num_channels+1] = filteredGreen[i];
        filteredImage[i*input_jpeg.num_channels+2] = filteredBlue[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();  // Stop recording time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output Gray JPEG Image
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
