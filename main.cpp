#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include "seam_carving_bonus.h"

// =======================================================================
// SEAM CARVING ALGORITHM IMPLEMENTATION
// =======================================================================

// Step 1: Calculate energy map using Sobel filter
cv::Mat computeEnergyMap(const cv::Mat& image) {
    // Convert to grayscale if needed
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }

    // Compute gradients using Sobel filter
    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, CV_32F, 1, 0, 3);  // X gradient
    cv::Sobel(gray, sobelY, CV_32F, 0, 1, 3);  // Y gradient

    // Compute magnitude of gradients (energy)
    cv::Mat energy(image.rows, image.cols, CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float dx = sobelX.at<float>(i, j);
            float dy = sobelY.at<float>(i, j);
            energy.at<float>(i, j) = std::sqrt(dx * dx + dy * dy);
        }
    }

    return energy;
}

// Step 2: Find minimum energy vertical seam using dynamic programming
std::vector<int> findVerticalSeam(const cv::Mat& energy) {
    int rows = energy.rows;
    int cols = energy.cols;

    // DP table: dp[i][j] = minimum cumulative energy to reach pixel (i,j)
    std::vector<std::vector<float>> dp(rows, std::vector<float>(cols, FLT_MAX));
    std::vector<std::vector<int>> parent(rows, std::vector<int>(cols, -1));

    // Initialize first row
    for (int j = 0; j < cols; j++) {
        dp[0][j] = energy.at<float>(0, j);
    }

    // Fill DP table from top to bottom
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Check three pixels above: (i-1, j-1), (i-1, j), (i-1, j+1)
            for (int k = j - 1; k <= j + 1; k++) {
                if (k >= 0 && k < cols) {
                    float candidate = dp[i - 1][k] + energy.at<float>(i, j);
                    if (candidate < dp[i][j]) {
                        dp[i][j] = candidate;
                        parent[i][j] = k;
                    }
                }
            }
        }
    }

    // Find minimum in last row
    int minCol = 0;
    float minEnergy = dp[rows - 1][0];
    for (int j = 1; j < cols; j++) {
        if (dp[rows - 1][j] < minEnergy) {
            minEnergy = dp[rows - 1][j];
            minCol = j;
        }
    }

    // Backtrack to get seam path
    std::vector<int> seam(rows);
    seam[rows - 1] = minCol;
    for (int i = rows - 2; i >= 0; i--) {
        seam[i] = parent[i + 1][seam[i + 1]];
    }

    return seam;
}

// Step 3: Remove vertical seam from image
cv::Mat removeVerticalSeam(const cv::Mat& image, const std::vector<int>& seam) {
    int rows = image.rows;
    int cols = image.cols;
    cv::Mat result(rows, cols - 1, image.type());

    for (int i = 0; i < rows; i++) {
        int seamCol = seam[i];
        for (int j = 0, k = 0; j < cols; j++) {
            if (j != seamCol) {
                if (image.channels() == 3) {
                    result.at<cv::Vec3b>(i, k++) = image.at<cv::Vec3b>(i, j);
                }
                else {
                    result.at<uint8_t>(i, k++) = image.at<uint8_t>(i, j);
                }
            }
        }
    }

    return result;
}

// Remove horizontal seam by rotating, removing vertical seam, and rotating back
cv::Mat removeHorizontalSeam(const cv::Mat& image) {
    // Rotate 90 degrees clockwise
    cv::Mat rotated;
    cv::transpose(image, rotated);
    cv::flip(rotated, rotated, 1);

    // Find and remove vertical seam
    cv::Mat energy = computeEnergyMap(rotated);
    std::vector<int> seam = findVerticalSeam(energy);
    rotated = removeVerticalSeam(rotated, seam);

    // Rotate back 90 degrees counterclockwise
    cv::flip(rotated, rotated, 1);  // Changed from flip(0) to flip(1)
    cv::transpose(rotated, rotated);

    return rotated;
}

// Main seam carving function (non-visualization version)
cv::Mat seamCarve(cv::Mat image, int targetWidth, int targetHeight) {
    std::cout << "Carving from " << image.cols << "x" << image.rows
        << " to " << targetWidth << "x" << targetHeight << "\n";

    // Remove vertical seams
    int verticalSeamsToRemove = image.cols - targetWidth;
    for (int i = 0; i < verticalSeamsToRemove; i++) {
        if ((i + 1) % 50 == 0 || i == 0) {
            std::cout << "  Vertical seam " << (i + 1) << "/" << verticalSeamsToRemove << "\n";
        }
        cv::Mat energy = computeEnergyMap(image);
        std::vector<int> seam = findVerticalSeam(energy);
        image = removeVerticalSeam(image, seam);
    }

    // Remove horizontal seams
    int horizontalSeamsToRemove = image.rows - targetHeight;
    for (int i = 0; i < horizontalSeamsToRemove; i++) {
        if ((i + 1) % 50 == 0 || i == 0) {
            std::cout << "  Horizontal seam " << (i + 1) << "/" << horizontalSeamsToRemove << "\n";
        }
        image = removeHorizontalSeam(image);
    }

    std::cout << "✓ Carving complete! Final size: " << image.cols << "x" << image.rows << "\n";
    return image;
}

// =======================================================================
// HELPER FUNCTION TO DISPLAY IMAGE SCALED TO FIT SCREEN
// =======================================================================

void displayImageScaled(const std::string& windowName, const cv::Mat& image, int maxWidth = 1400, int maxHeight = 800, bool interactive = false) {
    float scaleX = (float)maxWidth / image.cols;
    float scaleY = (float)maxHeight / image.rows;
    float scale = std::min(scaleX, scaleY);

    int displayWidth = (int)(image.cols * scale);
    int displayHeight = (int)(image.rows * scale);

    // Only resize if window exists (don't recreate it)
    cv::resizeWindow(windowName, displayWidth, displayHeight);
    cv::imshow(windowName, image);

    // If interactive, keep processing events so window is movable
    if (interactive) {
        std::cout << "\n[Window is fully interactive - you can move/resize it]\n";
        std::cout << "Press any key in the window or 'q' to continue to menu...\n";
        while (true) {
            int key = cv::waitKey(100);  // Wait for key with window event processing
            if (key != -1) {  // If key pressed (not -1)
                break;
            }
        }
    }
}

// Create side-by-side comparison of original and current image
cv::Mat createComparison(const cv::Mat& original, const cv::Mat& current) {
    // Resize both images to same height for side-by-side comparison
    int maxHeight = std::max(original.rows, current.rows);

    cv::Mat orig_resized, curr_resized;

    // Scale original to match height
    float origScale = (float)maxHeight / original.rows;
    cv::resize(original, orig_resized, cv::Size(
        (int)(original.cols * origScale),
        maxHeight
    ));

    // Scale current to match height
    float currScale = (float)maxHeight / current.rows;
    cv::resize(current, curr_resized, cv::Size(
        (int)(current.cols * currScale),
        maxHeight
    ));

    // Create side-by-side image
    int totalWidth = orig_resized.cols + curr_resized.cols + 20;  // 20px gap
    cv::Mat comparison(maxHeight, totalWidth, CV_8UC3, cv::Scalar(50, 50, 50));

    // Copy images
    orig_resized.copyTo(comparison(cv::Rect(0, 0, orig_resized.cols, maxHeight)));
    curr_resized.copyTo(comparison(cv::Rect(orig_resized.cols + 20, 0, curr_resized.cols, maxHeight)));

    // Add labels
    cv::putText(comparison, "ORIGINAL", cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
    cv::putText(comparison, "CURRENT", cv::Point(orig_resized.cols + 30, 30),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

    return comparison;
}

// =======================================================================
// MAIN APPLICATION (Enhanced with Bonus Features)
// =======================================================================

int main(int argc, char* argv[]) {
    std::cout << "\n+======================================+\n";
    std::cout << "|   SEAM CARVING APPLICATION          |\n";
    std::cout << "|   WITH BONUS FEATURES!              |\n";
    std::cout << "|   Working dir: " << std::filesystem::current_path().string().substr(0, 20) << "...|\n";
    std::cout << "+======================================+\n\n";

    // Load image
    cv::Mat originalImage = cv::imread("../../../Images/tower.jpg");

    if (originalImage.empty()) {
        std::cerr << "Error: Could not load image at ../../../Images/tower.jpg\n";
        return -1;
    }

    std::cout << "✓ Image loaded: " << originalImage.cols << "x" << originalImage.rows << " pixels\n\n";

    cv::Mat currentImage = originalImage.clone();
    bool running = true;

    // Create resizable window that fits on screen
    cv::namedWindow("Seam Carving Tool", cv::WINDOW_NORMAL | cv::WINDOW_GUI_NORMAL);
    cv::resizeWindow("Seam Carving Tool", 1200, 800);
    cv::moveWindow("Seam Carving Tool", 100, 100);

    // Visualization settings for bonus features
    VisualizationSettings visSettings;

    std::cout << "Window created at position (100, 100) with size 1200x800\n";
    std::cout << "You can now drag the window by its title bar!\n\n";

    displayImageScaled("Seam Carving Tool", currentImage);
    cv::pollKey();

    while (running) {
        std::cout << "\n+====================================+\n";
        std::cout << "|    SEAM CARVING MENU               |\n";
        std::cout << "|    Current: " << currentImage.cols << "x" << currentImage.rows << "          |\n";
        std::cout << "+====================================+\n";
        std::cout << "| BASIC OPERATIONS                   |\n";
        std::cout << "| [1] Custom size (width x height)   |\n";
        std::cout << "| [2] Reduce width by 100px          |\n";
        std::cout << "| [3] Reduce height by 100px         |\n";
        std::cout << "| [4] Reset to original              |\n";
        std::cout << "|                                    |\n";
        std::cout << "| BONUS FEATURES (STAR)              |\n";
        std::cout << "| [5] Seam carving WITH visualization|\n";
        std::cout << "| [6] Interactive mouse selection    |\n";
        std::cout << "| [7] Step-by-step mode              |\n";
        std::cout << "| [8] Preset aspect ratios           |\n";
        std::cout << "| [9] Visualization settings         |\n";
        std::cout << "|                                    |\n";
        std::cout << "| FILE OPERATIONS                    |\n";
        std::cout << "| [S] Save result                    |\n";
        std::cout << "| [V] Save with visualizations       |\n";
        std::cout << "| [Q] Exit                           |\n";
        std::cout << "+====================================+\n";
        std::cout << "Enter choice: ";

        std::string choice;
        std::cin >> choice;

        if (choice == "1") {
            int width, height;
            std::cout << "Enter target width (100-" << originalImage.cols << "): ";
            std::cin >> width;
            std::cout << "Enter target height (100-" << originalImage.rows << "): ";
            std::cin >> height;

            if (width < 100 || height < 100) {
                std::cout << "Error: Minimum size is 100x100\n";
                continue;
            }
            if (width > currentImage.cols || height > currentImage.rows) {
                std::cout << "Error: Can only reduce size, not enlarge\n";
                continue;
            }

            std::cout << "\nProcessing (this may take a moment)...\n";
            auto start = std::chrono::high_resolution_clock::now();

            currentImage = seamCarve(currentImage, width, height);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "Time taken: " << duration.count() << "ms\n";
            std::cout << "✓ Showing SIDE-BY-SIDE comparison (Original | Current)\n";
            cv::Mat comparison = createComparison(originalImage, currentImage);
            displayImageScaled("Seam Carving Tool", comparison, 1600, 900, true);
        }
        else if (choice == "2") {
            int newWidth = std::max(100, currentImage.cols - 100);
            std::cout << "\nProcessing...\n";
            currentImage = seamCarve(currentImage, newWidth, currentImage.rows);
            std::cout << "✓ Showing SIDE-BY-SIDE comparison (Original | Current)\n";
            cv::Mat comparison = createComparison(originalImage, currentImage);
            displayImageScaled("Seam Carving Tool", comparison, 1600, 900, true);
        }
        else if (choice == "3") {
            int newHeight = std::max(100, currentImage.rows - 100);
            std::cout << "\nProcessing...\n";
            currentImage = seamCarve(currentImage, currentImage.cols, newHeight);
            std::cout << "✓ Showing SIDE-BY-SIDE comparison (Original | Current)\n";
            cv::Mat comparison = createComparison(originalImage, currentImage);
            displayImageScaled("Seam Carving Tool", comparison, 1600, 900, true);
        }
        else if (choice == "4") {
            currentImage = originalImage.clone();
            std::cout << "✓ Reset to original (" << currentImage.cols << "x" << currentImage.rows << ")\n";
            displayImageScaled("Seam Carving Tool", currentImage, 1600, 900, true);
        }
        // ===== BONUS FEATURES =====
        else if (choice == "5") {
            int width, height;
            std::cout << "Enter target width (100-" << currentImage.cols << "): ";
            std::cin >> width;
            std::cout << "Enter target height (100-" << currentImage.rows << "): ";
            std::cin >> height;

            if (width < 100 || height < 100) {
                std::cout << "Error: Minimum size is 100x100\n";
                continue;
            }
            if (width > currentImage.cols || height > currentImage.rows) {
                std::cout << "Error: Can only reduce size, not enlarge\n";
                continue;
            }

            std::cout << "\nStarting INTERACTIVE visualization mode...\n";
            auto start = std::chrono::high_resolution_clock::now();

            currentImage = seamCarveWithVisualization(currentImage, width, height,
                "Seam Carving Tool", visSettings);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "Time taken: " << duration.count() << "ms\n";
            cv::Mat comparison = createComparison(originalImage, currentImage);
            displayImageScaled("Seam Carving Tool", comparison, 1600, 900, true);
        }
        else if (choice == "6") {
            std::cout << "\n✓ Entering MOUSE SELECTION mode...\n";
            displayImageScaled("Seam Carving Tool", currentImage, 1400, 800, false);

            cv::Rect selection = interactiveMouseSelection(currentImage, "Seam Carving Tool");

            if (selection.width > 0 && selection.height > 0) {
                std::cout << "Selected dimensions: " << selection.width << "x" << selection.height << "\n";
                std::cout << "Apply seam carving to these dimensions? (y/n): ";
                char confirm;
                std::cin >> confirm;

                if (confirm == 'y' || confirm == 'Y') {
                    std::cout << "\nStarting visualization...\n";
                    currentImage = seamCarveWithVisualization(currentImage, selection.width,
                        selection.height,
                        "Seam Carving Tool", visSettings);
                    cv::Mat comparison = createComparison(originalImage, currentImage);
                    displayImageScaled("Seam Carving Tool", comparison, 1600, 900, true);
                }
            }
            else {
                std::cout << "Selection cancelled.\n";
            }
        }
        else if (choice == "7") {
            std::cout << "\n✓ Entering STEP-BY-STEP mode...\n";
            displayImageScaled("Seam Carving Tool", currentImage, 1400, 800, false);
            runStepByStepMode(currentImage, "Seam Carving Tool");
            std::cout << "Exited step-by-step mode.\n";
        }
        else if (choice == "8") {
            std::cout << "\nPRESET ASPECT RATIOS:\n";
            std::cout << "[1] 16:9 (Widescreen)\n";
            std::cout << "[2] 4:3 (Standard)\n";
            std::cout << "[3] 1:1 (Square)\n";
            std::cout << "[4] 50% reduction\n";
            std::cout << "Choose preset: ";

            int preset;
            std::cin >> preset;

            int targetWidth, targetHeight;
            if (calculatePresetDimensions(currentImage.cols, currentImage.rows, preset,
                targetWidth, targetHeight)) {
                std::cout << "Target dimensions: " << targetWidth << "x" << targetHeight << "\n";
                std::cout << "Apply with visualization? (y/n): ";
                char confirm;
                std::cin >> confirm;

                if (confirm == 'y' || confirm == 'Y') {
                    currentImage = seamCarveWithVisualization(currentImage, targetWidth,
                        targetHeight,
                        "Seam Carving Tool", visSettings);
                    cv::Mat comparison = createComparison(originalImage, currentImage);
                    displayImageScaled("Seam Carving Tool", comparison, 1600, 900, true);
                }
            }
            else {
                std::cout << "Invalid preset!\n";
            }
        }
        else if (choice == "9") {
            handleVisualizationSettings(visSettings);
        }
        // ===== FILE OPERATIONS =====
        else if (choice == "s" || choice == "S") {
            std::string filename = "output_" + std::to_string(currentImage.cols) + "x" +
                std::to_string(currentImage.rows) + ".jpg";
            if (cv::imwrite(filename, currentImage)) {
                std::cout << "✓ Saved to: " << filename << "\n";
            }
            else {
                std::cout << "✗ Error: Could not save file\n";
            }
        }
        else if (choice == "v" || choice == "V") {
            std::cout << "Enter base filename (without extension): ";
            std::string baseFilename;
            std::cin >> baseFilename;

            if (saveWithVisualizationExamples(currentImage, baseFilename)) {
                std::cout << "✓ All files saved successfully!\n";
            }
            else {
                std::cout << "✗ Error saving files\n";
            }
        }
        else if (choice == "q" || choice == "Q") {
            running = false;
           
        }
        else {
            std::cout << "✗ Invalid choice!\n";
        }
    }

    cv::destroyAllWindows();
    return 0;
}