#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <opencv2/opencv.hpp> // Includes all necessary OpenCV headers
#include <iostream>
#include <vector>
#include <filesystem>
int main() {
    std::cout << "Working dir: " << std::filesystem::current_path() << "\n";
    // 1. Load an image from a file.
        // Make sure you have an image file (e.g., "castle.png") in the same
        // directory as your executable, or provide the full path.
    cv::Mat image = cv::imread("../../../Images/ghibli.jpg");

    // 2. Check if the image was loaded successfully.
    if (image.empty()) {
        std::cout << "Error: Could not load the image." << std::endl;
        return -1;
    }

    // 3. Display the image in a window.
    // The first argument is the window title, the second is the image data.
    cv::imshow("My Image Display", image);

    // 4. Wait for a key press.
    // This function is required to process GUI events. The window will close
    // as soon as you press any key. The 0 means it waits indefinitely.
    cv::waitKey(0);

    return 0;
}