#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <chrono>

// ═══════════════════════════════════════════════════════════════════════════
// ENERGY MAP COMPUTATION (SHARED)
// ═══════════════════════════════════════════════════════════════════════════

cv::Mat computeEnergyMap(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }

    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sobelY, CV_32F, 0, 1, 3);

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

// ═══════════════════════════════════════════════════════════════════════════
// DYNAMIC PROGRAMMING SEAM FINDING
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> findVerticalSeamDP(const cv::Mat& energy) {
    int rows = energy.rows;
    int cols = energy.cols;

    std::vector<std::vector<float>> dp(rows, std::vector<float>(cols, FLT_MAX));
    std::vector<std::vector<int>> parent(rows, std::vector<int>(cols, -1));

    // Initialize first row
    for (int j = 0; j < cols; j++) {
        dp[0][j] = energy.at<float>(0, j);
    }

    // Fill DP table
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
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

    // Backtrack
    std::vector<int> seam(rows);
    seam[rows - 1] = minCol;
    for (int i = rows - 2; i >= 0; i--) {
        seam[i] = parent[i + 1][seam[i + 1]];
    }

    return seam;
}

// ═══════════════════════════════════════════════════════════════════════════
// GREEDY SEAM FINDING
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> findVerticalSeamGreedy(const cv::Mat& energy) {
    int rows = energy.rows;
    int cols = energy.cols;

    std::vector<int> seam(rows);

    // Start from minimum in first row
    int currentCol = 0;
    float minEnergy = energy.at<float>(0, 0);
    for (int j = 1; j < cols; j++) {
        if (energy.at<float>(0, j) < minEnergy) {
            minEnergy = energy.at<float>(0, j);
            currentCol = j;
        }
    }
    seam[0] = currentCol;

    // Greedy: at each row, pick minimum among three neighbors
    for (int i = 1; i < rows; i++) {
        int bestCol = currentCol;
        float bestEnergy = FLT_MAX;

        for (int k = currentCol - 1; k <= currentCol + 1; k++) {
            if (k >= 0 && k < cols) {
                float energyValue = energy.at<float>(i, k);
                if (energyValue < bestEnergy) {
                    bestEnergy = energyValue;
                    bestCol = k;
                }
            }
        }

        seam[i] = bestCol;
        currentCol = bestCol;
    }

    return seam;
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

float calculateSeamEnergy(const cv::Mat& energy, const std::vector<int>& seam) {
    float totalEnergy = 0.0f;
    for (size_t i = 0; i < seam.size(); i++) {
        totalEnergy += energy.at<float>(i, seam[i]);
    }
    return totalEnergy;
}

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

// Visualize a seam on an image
cv::Mat visualizeSeam(const cv::Mat& image, const std::vector<int>& seam, cv::Scalar color) {
    cv::Mat result = image.clone();
    for (size_t i = 0; i < seam.size(); i++) {
        int col = seam[i];
        // Draw a thicker line for visibility
        for (int offset = -1; offset <= 1; offset++) {
            if (col + offset >= 0 && col + offset < result.cols) {
                result.at<cv::Vec3b>(i, col + offset) = cv::Vec3b(color[0], color[1], color[2]);
            }
        }
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPARISON FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

struct AlgorithmStats {
    long long timeTaken;  // milliseconds
    float totalSeamEnergy;
    int seamsRemoved;
};

AlgorithmStats seamCarveWithStats(cv::Mat& image, int targetWidth,
    bool useDP, const std::string& algorithmName) {
    AlgorithmStats stats = { 0, 0.0f, 0 };

    int seamsToRemove = image.cols - targetWidth;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < seamsToRemove; i++) {
        cv::Mat energy = computeEnergyMap(image);
        std::vector<int> seam;

        if (useDP) {
            seam = findVerticalSeamDP(energy);
        }
        else {
            seam = findVerticalSeamGreedy(energy);
        }

        stats.totalSeamEnergy += calculateSeamEnergy(energy, seam);
        stats.seamsRemoved++;

        image = removeVerticalSeam(image, seam);
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats.timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return stats;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPARISON APPLICATION
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     DP vs GREEDY SEAM CARVING COMPARISON TOOL            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    // Load image
    cv::Mat originalImage = cv::imread("../../../Images/Broadway_tower_edit.jpg");
    if (originalImage.empty()) {
        std::cerr << "Error: Could not load image\n";
        return -1;
    }

    std::cout << "✓ Image loaded: " << originalImage.cols << "x" << originalImage.rows << " pixels\n\n";

    // Get target dimensions
    int targetWidth;
    std::cout << "Enter target width (smaller than " << originalImage.cols << "): ";
    std::cin >> targetWidth;

    if (targetWidth >= originalImage.cols || targetWidth < 100) {
        std::cerr << "Invalid width!\n";
        return -1;
    }

    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Running DYNAMIC PROGRAMMING algorithm...\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    cv::Mat dpImage = originalImage.clone();
    AlgorithmStats dpStats = seamCarveWithStats(dpImage, targetWidth, true, "DP");

    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Running GREEDY algorithm...\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    cv::Mat greedyImage = originalImage.clone();
    AlgorithmStats greedyStats = seamCarveWithStats(greedyImage, targetWidth, false, "Greedy");

    // Print comparison results
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    COMPARISON RESULTS                     ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Metric                    │ DP          │ Greedy         ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";

    printf("║ Time (ms)                 │ %-11lld │ %-14lld ║\n",
        dpStats.timeTaken, greedyStats.timeTaken);

    float dpAvgEnergy = dpStats.totalSeamEnergy / dpStats.seamsRemoved;
    float greedyAvgEnergy = greedyStats.totalSeamEnergy / greedyStats.seamsRemoved;

    printf("║ Avg Seam Energy           │ %-11.2f │ %-14.2f ║\n",
        dpAvgEnergy, greedyAvgEnergy);

    printf("║ Total Energy              │ %-11.2f │ %-14.2f ║\n",
        dpStats.totalSeamEnergy, greedyStats.totalSeamEnergy);

    float energyDiff = ((greedyStats.totalSeamEnergy - dpStats.totalSeamEnergy) / dpStats.totalSeamEnergy) * 100;
    printf("║ Energy Difference         │ Baseline    │ +%.2f%%        ║\n", energyDiff);

    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    // Show one more comparison: find the same seam with both algorithms
    std::cout << "Visual comparison of a single seam:\n";
    cv::Mat energy = computeEnergyMap(originalImage);
    std::vector<int> dpSeam = findVerticalSeamDP(energy);
    std::vector<int> greedySeam = findVerticalSeamGreedy(energy);

    float dpSeamEnergy = calculateSeamEnergy(energy, dpSeam);
    float greedySeamEnergy = calculateSeamEnergy(energy, greedySeam);

    std::cout << "  DP Seam Energy:     " << dpSeamEnergy << " (optimal)\n";
    std::cout << "  Greedy Seam Energy: " << greedySeamEnergy;
    if (greedySeamEnergy > dpSeamEnergy) {
        std::cout << " (+" << ((greedySeamEnergy - dpSeamEnergy) / dpSeamEnergy * 100) << "% worse)\n";
    }
    else {
        std::cout << " (same as optimal)\n";
    }

    // Visualize seams
    cv::Mat dpSeamVis = visualizeSeam(originalImage, dpSeam, cv::Scalar(0, 255, 0));    // Green
    cv::Mat greedySeamVis = visualizeSeam(originalImage, greedySeam, cv::Scalar(0, 0, 255)); // Red

    // Create comparison image
    int maxHeight = originalImage.rows;
    int totalWidth = dpSeamVis.cols + greedySeamVis.cols + 20;
    cv::Mat seamComparison(maxHeight, totalWidth, CV_8UC3, cv::Scalar(50, 50, 50));

    dpSeamVis.copyTo(seamComparison(cv::Rect(0, 0, dpSeamVis.cols, maxHeight)));
    greedySeamVis.copyTo(seamComparison(cv::Rect(dpSeamVis.cols + 20, 0, greedySeamVis.cols, maxHeight)));

    cv::putText(seamComparison, "DP (Green)", cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::putText(seamComparison, "GREEDY (Red)", cv::Point(dpSeamVis.cols + 30, 30),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

    // Create result comparison
    cv::Mat resultComparison(maxHeight, totalWidth, CV_8UC3, cv::Scalar(50, 50, 50));
    dpImage.copyTo(resultComparison(cv::Rect(0, 0, dpImage.cols, maxHeight)));
    greedyImage.copyTo(resultComparison(cv::Rect(dpImage.cols + 20, 0, greedyImage.cols, maxHeight)));

    cv::putText(resultComparison, "DP RESULT", cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
    cv::putText(resultComparison, "GREEDY RESULT", cv::Point(dpImage.cols + 30, 30),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

    // Display results
    cv::namedWindow("Single Seam Comparison", cv::WINDOW_NORMAL);
    cv::resizeWindow("Single Seam Comparison", 1600, 800);
    cv::imshow("Single Seam Comparison", seamComparison);

    cv::namedWindow("Final Results Comparison", cv::WINDOW_NORMAL);
    cv::resizeWindow("Final Results Comparison", 1600, 800);
    cv::imshow("Final Results Comparison", resultComparison);

    // Save results
    cv::imwrite("dp_result.jpg", dpImage);
    cv::imwrite("greedy_result.jpg", greedyImage);
    cv::imwrite("seam_comparison.jpg", seamComparison);
    cv::imwrite("final_comparison.jpg", resultComparison);

    std::cout << "\n✓ Results saved to: dp_result.jpg, greedy_result.jpg\n";
    std::cout << "✓ Comparisons saved to: seam_comparison.jpg, final_comparison.jpg\n";
    std::cout << "\nPress any key to exit...\n";
    cv::waitKey(0);

    return 0;
}