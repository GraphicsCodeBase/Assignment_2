#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>

// =======================================================================
// GREEDY SEAM CARVING ALGORITHM IMPLEMENTATION
// =======================================================================

// Step 1: Calculate energy map using Sobel filter (same as DP version)
// Stronger energy: sum of color-channel gradients + a small Laplacian texture term
cv::Mat computeEnergyMap(const cv::Mat& image) {
    CV_Assert(!image.empty());
    cv::Mat img32f;
    if (image.type() != CV_8UC3) {
        cv::Mat tmp;
        if (image.channels() == 1) cv::cvtColor(image, tmp, cv::COLOR_GRAY2BGR);
        else tmp = image;
        tmp.convertTo(img32f, CV_32FC3, 1.0 / 255.0);
    }
    else {
        image.convertTo(img32f, CV_32FC3, 1.0 / 255.0);
    }

    // Per-channel Sobel, then sum squared magnitudes across channels
    std::vector<cv::Mat> ch; cv::split(img32f, ch);
    cv::Mat energy = cv::Mat::zeros(image.size(), CV_32F);

    for (int c = 0; c < 3; ++c) {
        cv::Mat gx, gy;
        // Scharr is a bit sharper than Sobel; use Sobel(3) if you prefer
        cv::Scharr(ch[c], gx, CV_32F, 1, 0);
        cv::Scharr(ch[c], gy, CV_32F, 0, 1);
        energy += gx.mul(gx) + gy.mul(gy);
    }
    cv::sqrt(energy, energy);

    // Small texture boost so fine details (like a person/towel) get protected
    cv::Mat gray, lap, absLap;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_32F, 3);
    cv::absdiff(lap, 0, absLap);                // |Laplacian|
    energy += 0.20f * absLap;                   // tune 0.10–0.35

    // Optional: light blur to stabilize noisy gradients
    cv::GaussianBlur(energy, energy, cv::Size(3, 3), 0);

    return energy;
}


// Step 2: Find vertical seam using PURE GREEDY ALGORITHM
// At each row, simply pick the minimum energy pixel among the three valid neighbors
// This is a LOCAL decision - only considers current pixel energy, not future rows
std::vector<int> findVerticalSeamGreedy(const cv::Mat& energy) {
    const int rows = energy.rows, cols = energy.cols;
    std::vector<int> seam(rows);

    // Start at global min in top row
    int currentCol = 0;
    float best = energy.at<float>(0, 0);
    for (int j = 1; j < cols; ++j) {
        float v = energy.at<float>(0, j);
        if (v < best) {
            best = v;
            currentCol = j;
        }
    }
    seam[0] = currentCol;

    // For each subsequent row, pick the minimum energy neighbor
    // This is the GREEDY part - only looks at current row, not ahead
    for (int i = 1; i < rows; ++i) {
        int bestCol = currentCol;
        float minEnergy = std::numeric_limits<float>::infinity();

        // Check the three valid neighbors from current position
        for (int k = currentCol - 1; k <= currentCol + 1; ++k) {
            if (k >= 0 && k < cols) {
                float e = energy.at<float>(i, k);  // Only current pixel energy!
                if (e < minEnergy) {
                    minEnergy = e;
                    bestCol = k;
                }
            }
        }

        seam[i] = bestCol;
        currentCol = bestCol;
    }

    return seam;
}



// Calculate total energy of a seam (for comparison purposes)
float calculateSeamEnergy(const cv::Mat& energy, const std::vector<int>& seam) {
    float totalEnergy = 0.0f;
    for (int i = 0; i < seam.size(); i++) {
        totalEnergy += energy.at<float>(i, seam[i]);
    }
    return totalEnergy;
}

// Step 3: Remove vertical seam from image (same as DP version)
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

    // Find and remove vertical seam using GREEDY algorithm
    cv::Mat energy = computeEnergyMap(rotated);
    std::vector<int> seam = findVerticalSeamGreedy(energy);
    rotated = removeVerticalSeam(rotated, seam);

    // Rotate back 90 degrees counterclockwise
    cv::flip(rotated, rotated, 0);
    cv::transpose(rotated, rotated);

    return rotated;
}

// Main seam carving function using GREEDY algorithm
cv::Mat seamCarveGreedy(cv::Mat image, int targetWidth, int targetHeight) {
    std::cout << "Carving (GREEDY) from " << image.cols << "x" << image.rows
        << " to " << targetWidth << "x" << targetHeight << "\n";

    float totalSeamEnergy = 0.0f;
    int seamCount = 0;

    // Remove vertical seams
    int verticalSeamsToRemove = image.cols - targetWidth;
    for (int i = 0; i < verticalSeamsToRemove; i++) {
        if ((i + 1) % 50 == 0 || i == 0) {
            std::cout << "  Vertical seam " << (i + 1) << "/" << verticalSeamsToRemove << "\n";
        }
        cv::Mat energy = computeEnergyMap(image);
        std::vector<int> seam = findVerticalSeamGreedy(energy);

        // Track seam energy for statistics
        totalSeamEnergy += calculateSeamEnergy(energy, seam);
        seamCount++;

        image = removeVerticalSeam(image, seam);
    }

    // Remove horizontal seams
    int horizontalSeamsToRemove = image.rows - targetHeight;
    for (int i = 0; i < horizontalSeamsToRemove; i++) {
        if ((i + 1) % 50 == 0 || i == 0) {
            std::cout << "  Horizontal seam " << (i + 1) << "/" << horizontalSeamsToRemove << "\n";
        }
        image = removeHorizontalSeam(image);
        seamCount++;
    }

    if (seamCount > 0) {
        std::cout << "✓ Average seam energy: " << (totalSeamEnergy / seamCount) << "\n";
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
        std::cout << "Press any key in the window to continue to menu...\n";
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
    cv::putText(comparison, "GREEDY RESULT", cv::Point(orig_resized.cols + 30, 30),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

    return comparison;
}

// =======================================================================
// MAIN APPLICATION
// =======================================================================

int main(int argc, char* argv[]) {
    std::cout << "GREEDY SEAM CARVING APPLICATION    \n";
    std::cout << "Working dir: " << std::filesystem::current_path().string().substr(0, 24) << "...|\n";

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
    cv::namedWindow("Greedy Seam Carving Tool", cv::WINDOW_NORMAL | cv::WINDOW_GUI_NORMAL);
    cv::resizeWindow("Greedy Seam Carving Tool", 1200, 800);
    cv::moveWindow("Greedy Seam Carving Tool", 100, 100);

    std::cout << "Window created - using GREEDY ALGORITHM\n\n";

    displayImageScaled("Greedy Seam Carving Tool", currentImage);
    cv::pollKey();

    while (running) {
        std::cout << "    GREEDY SEAM CARVING MENU           \n";
        std::cout << "    Current: " << currentImage.cols << "x" << currentImage.rows << "            |\n";
        std::cout << "[1] Custom size (width x height)      \n";
        std::cout << "[2] Reduce width by 100px             \n";
        std::cout << "[3] Reduce height by 100px            \n";
        std::cout << "[4] Reset to original                 \n";
        std::cout << "[5] Save result                       \n";
        std::cout << "[6] Exit                              \n";
        std::cout << "Note: Using GREEDY algorithm (local decisions)\n";
        std::cout << "Enter choice (1-6): ";

        int choice;
        std::cin >> choice;

        switch (choice) {
        case 1: {
            int width, height;
            std::cout << "Enter target width (100-" << originalImage.cols << "): ";
            std::cin >> width;
            std::cout << "Enter target height (100-" << originalImage.rows << "): ";
            std::cin >> height;

            if (width < 100 || height < 100) {
                std::cout << "Error: Minimum size is 100x100\n";
                break;
            }
            if (width > currentImage.cols || height > currentImage.rows) {
                std::cout << "Error: Can only reduce size, not enlarge\n";
                break;
            }

            std::cout << "\nProcessing with GREEDY algorithm...\n";
            auto start = std::chrono::high_resolution_clock::now();

            currentImage = seamCarveGreedy(currentImage, width, height);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "Time taken: " << duration.count() << "ms\n";
            std::cout << "✓ Showing SIDE-BY-SIDE comparison (Original | Greedy Result)\n";
            cv::Mat comparison = createComparison(originalImage, currentImage);
            displayImageScaled("Greedy Seam Carving Tool", comparison, 1600, 900, true);
            break;
        }
        case 2: {
            int newWidth = std::max(100, currentImage.cols - 100);
            std::cout << "\nProcessing with GREEDY algorithm...\n";
            currentImage = seamCarveGreedy(currentImage, newWidth, currentImage.rows);
            std::cout << "✓ Showing SIDE-BY-SIDE comparison (Original | Greedy Result)\n";
            cv::Mat comparison = createComparison(originalImage, currentImage);
            displayImageScaled("Greedy Seam Carving Tool", comparison, 1600, 900, true);
            break;
        }
        case 3: {
            int newHeight = std::max(100, currentImage.rows - 100);
            std::cout << "\nProcessing with GREEDY algorithm...\n";
            currentImage = seamCarveGreedy(currentImage, currentImage.cols, newHeight);
            std::cout << "✓ Showing SIDE-BY-SIDE comparison (Original | Greedy Result)\n";
            cv::Mat comparison = createComparison(originalImage, currentImage);
            displayImageScaled("Greedy Seam Carving Tool", comparison, 1600, 900, true);
            break;
        }
        case 4: {
            currentImage = originalImage.clone();
            std::cout << "✓ Reset to original (" << currentImage.cols << "x" << currentImage.rows << ")\n";
            displayImageScaled("Greedy Seam Carving Tool", currentImage, 1600, 900, true);
            break;
        }
        case 5: {
            std::string filename = "greedy_output_" + std::to_string(currentImage.cols) + "x" +
                std::to_string(currentImage.rows) + ".jpg";
            if (cv::imwrite(filename, currentImage)) {
                std::cout << "✓ Saved to: " << filename << "\n";
            }
            else {
                std::cout << "✗ Error: Could not save file\n";
            }
            break;
        }
        case 6: {
            running = false;
            std::cout << "\n✓ Goodbye!\n";
            break;
        }
        default: {
            std::cout << "✗ Invalid choice! Enter 1-6.\n";
        }
        }
    }

    cv::destroyAllWindows();
    return 0;
}