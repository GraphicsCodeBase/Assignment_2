#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <fstream>

// =======================================================================
// ENERGY MAP COMPUTATION (Shared by DP & Greedy)
// =======================================================================

cv::Mat computeEnergyMap(const cv::Mat& image) {
    // Convert to grayscale if needed
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
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

// =======================================================================
// DYNAMIC PROGRAMMING SEAM FINDING
// =======================================================================

std::vector<int> findVerticalSeamDP(const cv::Mat& energy) {
    int rows = energy.rows, cols = energy.cols;
    std::vector<std::vector<float>> dp(rows, std::vector<float>(cols, FLT_MAX));
    std::vector<std::vector<int>> parent(rows, std::vector<int>(cols, -1));

    for (int j = 0; j < cols; j++) dp[0][j] = energy.at<float>(0, j);

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

    int minCol = 0;
    float minVal = dp[rows - 1][0];
    for (int j = 1; j < cols; j++) {
        if (dp[rows - 1][j] < minVal) { minVal = dp[rows - 1][j]; minCol = j; }
    }

    std::vector<int> seam(rows);
    seam[rows - 1] = minCol;
    for (int i = rows - 2; i >= 0; i--) seam[i] = parent[i + 1][seam[i + 1]];
    return seam;
}

// =======================================================================
// GREEDY SEAM FINDING (Pure Greedy - Local Decisions Only)
// =======================================================================

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
        float minEnergy = FLT_MAX;

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

// =======================================================================
// COMMON UTILITIES
// =======================================================================

float calculateSeamEnergy(const cv::Mat& energy, const std::vector<int>& seam) {
    float total = 0.f;
    for (size_t i = 0; i < seam.size(); ++i)
        total += energy.at<float>((int)i, seam[i]);
    return total;
}

cv::Mat removeVerticalSeam(const cv::Mat& image, const std::vector<int>& seam) {
    int rows = image.rows, cols = image.cols;
    cv::Mat result(rows, cols - 1, image.type());
    for (int i = 0; i < rows; i++) {
        int seamCol = seam[i];
        for (int j = 0, k = 0; j < cols; j++) {
            if (j == seamCol) continue;
            if (image.channels() == 3)
                result.at<cv::Vec3b>(i, k++) = image.at<cv::Vec3b>(i, j);
            else
                result.at<uchar>(i, k++) = image.at<uchar>(i, j);
        }
    }
    return result;
}

cv::Mat visualizeSeam(const cv::Mat& img, const std::vector<int>& seam, cv::Scalar color) {
    cv::Mat vis = img.clone();
    for (size_t i = 0; i < seam.size(); i++) {
        int c = seam[i];
        for (int dx = -1; dx <= 1; dx++)
            if (c + dx >= 0 && c + dx < vis.cols)
                vis.at<cv::Vec3b>((int)i, c + dx) = cv::Vec3b(color[0], color[1], color[2]);
    }
    return vis;
}

// =======================================================================
// SEAM CARVING COMPARISON LOGIC
// =======================================================================

struct Stats {
    long long timeMs;                    // Total execution time (milliseconds)
    float totalEnergy;                   // Sum of all seam energies
    int seamsRemoved;                    // Number of seams removed
    float avgSeamEnergy = 0.f;           // Average energy per seam
    float minSeamEnergy = FLT_MAX;       // Minimum seam energy
    float maxSeamEnergy = 0.f;           // Maximum seam energy
    long long energyMapTimeMs = 0;       // Time spent computing energy maps
};

Stats seamCarveWithStats(cv::Mat& img, int targetWidth, int targetHeight, bool useDP) {
    Stats s{ 0, 0.f, 0 };
    auto totalStart = std::chrono::high_resolution_clock::now();

    // REMOVE VERTICAL SEAMS (to change width)
    int verticalSeams = img.cols - targetWidth;
    for (int i = 0; i < verticalSeams; i++) {
        // Time energy map computation
        auto energyStart = std::chrono::high_resolution_clock::now();
        cv::Mat energy = computeEnergyMap(img);
        auto energyEnd = std::chrono::high_resolution_clock::now();
        s.energyMapTimeMs += std::chrono::duration_cast<std::chrono::milliseconds>(
            energyEnd - energyStart).count();

        // Find seam (DP or Greedy)
        std::vector<int> seam = useDP ?
            findVerticalSeamDP(energy) : findVerticalSeamGreedy(energy);

        // Track seam energy statistics
        float seamEnergy = calculateSeamEnergy(energy, seam);
        s.totalEnergy += seamEnergy;
        s.minSeamEnergy = std::min(s.minSeamEnergy, seamEnergy);
        s.maxSeamEnergy = std::max(s.maxSeamEnergy, seamEnergy);
        s.seamsRemoved++;

        // Remove seam from image
        img = removeVerticalSeam(img, seam);
    }

    // REMOVE HORIZONTAL SEAMS (to change height)
    int horizontalSeams = img.rows - targetHeight;
    for (int i = 0; i < horizontalSeams; i++) {
        // Time energy map computation
        auto energyStart = std::chrono::high_resolution_clock::now();

        // Rotate image 90 degrees to treat horizontal seams as vertical
        cv::Mat rotated;
        cv::transpose(img, rotated);
        cv::flip(rotated, rotated, 1);

        cv::Mat energy = computeEnergyMap(rotated);
        auto energyEnd = std::chrono::high_resolution_clock::now();
        s.energyMapTimeMs += std::chrono::duration_cast<std::chrono::milliseconds>(
            energyEnd - energyStart).count();

        // Find seam (DP or Greedy) on rotated image
        std::vector<int> seam = useDP ?
            findVerticalSeamDP(energy) : findVerticalSeamGreedy(energy);

        // Track seam energy statistics
        float seamEnergy = calculateSeamEnergy(energy, seam);
        s.totalEnergy += seamEnergy;
        s.minSeamEnergy = std::min(s.minSeamEnergy, seamEnergy);
        s.maxSeamEnergy = std::max(s.maxSeamEnergy, seamEnergy);
        s.seamsRemoved++;

        // Remove seam from rotated image
        rotated = removeVerticalSeam(rotated, seam);

        // Rotate back
        cv::flip(rotated, rotated, 1);
        cv::transpose(rotated, img);
    }

    // Calculate total time and average energy
    s.timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - totalStart).count();
    s.avgSeamEnergy = s.seamsRemoved > 0 ? s.totalEnergy / s.seamsRemoved : 0.f;

    return s;
}

// =======================================================================
// DETAILED BENCHMARKING REPORT PRINTING
// =======================================================================

void printDetailedStats(const Stats& dp, const Stats& gr, int originalWidth, int originalHeight, int targetWidth, int targetHeight) {
    std::cout << "\n";
    std::cout << "+==================================================================+\n";
    std::cout << "|    DP vs GREEDY ALGORITHM - DETAILED PERFORMANCE ANALYSIS      |\n";
    std::cout << "+==================================================================+\n";

    // Test Configuration
    std::cout << "\nTEST CONFIGURATION:\n";
    std::cout << "  Original image:        " << originalWidth << "x" << originalHeight << " pixels\n";
    std::cout << "  Target image:          " << targetWidth << "x" << targetHeight << " pixels\n";
    std::cout << "  Vertical seams removed: " << (originalWidth - targetWidth) << "\n";
    std::cout << "  Horizontal seams removed: " << (originalHeight - targetHeight) << "\n";
    std::cout << "  Total seams removed:   " << dp.seamsRemoved << "\n\n";

    // Execution Time Comparison
    std::cout << "+--------- EXECUTION TIME COMPARISON ---------+\n";
    std::cout << std::left << std::setw(25) << "  DP Algorithm"
              << std::right << std::setw(15) << dp.timeMs << " ms\n";
    std::cout << std::left << std::setw(25) << "  Greedy Algorithm"
              << std::right << std::setw(15) << gr.timeMs << " ms\n";
    std::cout << std::left << std::setw(25) << "  Difference"
              << std::right << std::setw(15) << (dp.timeMs - gr.timeMs) << " ms\n";

    float speedup = (float)dp.timeMs / gr.timeMs;
    std::cout << "  Speedup Factor: Greedy is " << std::fixed << std::setprecision(2)
              << speedup << "x " << (speedup > 1 ? "faster" : "slower") << "\n\n";

    // Energy Statistics Table
    std::cout << "+------------ SEAM ENERGY STATISTICS -----------+\n";
    std::cout << std::left << std::setw(30) << "Metric";
    std::cout << std::right << std::setw(18) << "DP";
    std::cout << std::right << std::setw(18) << "Greedy\n";
    std::cout << std::string(66, '-') << "\n";

    printf("%-30s %16.2f %16.2f\n", "Total Energy:", dp.totalEnergy, gr.totalEnergy);
    printf("%-30s %16.2f %16.2f\n", "Average per Seam:", dp.avgSeamEnergy, gr.avgSeamEnergy);
    printf("%-30s %16.2f %16.2f\n", "Minimum Seam Energy:", dp.minSeamEnergy, gr.minSeamEnergy);
    printf("%-30s %16.2f %16.2f\n", "Maximum Seam Energy:", dp.maxSeamEnergy, gr.maxSeamEnergy);

    std::cout << "\n";

    // Quality Analysis
    float qualityDiff = gr.avgSeamEnergy - dp.avgSeamEnergy;
    float qualityPercent = (qualityDiff / dp.avgSeamEnergy) * 100.0f;

    std::cout << "+----------- QUALITY ANALYSIS ----------+\n";
    printf("  Quality Gap: %.2f%% ", qualityPercent);

    if (qualityPercent > 0) {
        std::cout << "(Greedy seams have higher energy - lower quality)\n";
        std::cout << "  ✓ DP produces better quality seams\n";
        std::cout << "  ✗ Greedy produces lower quality seams\n";
    } else {
        std::cout << "(Greedy better - unusual)\n";
    }

    // Time Breakdown
    std::cout << "\n+------ TIME BREAKDOWN (% of Total) ------+\n";
    float dpEnergyPercent = (dp.energyMapTimeMs / (float)dp.timeMs) * 100.0f;
    float dpAlgoPercent = 100.0f - dpEnergyPercent;
    float grEnergyPercent = (gr.energyMapTimeMs / (float)gr.timeMs) * 100.0f;
    float grAlgoPercent = 100.0f - grEnergyPercent;

    printf("  DP Algorithm:\n");
    printf("    Energy computation: %6.1f%% (%4lld ms)\n", dpEnergyPercent, dp.energyMapTimeMs);
    printf("    Seam finding:       %6.1f%% (%4lld ms)\n", dpAlgoPercent, dp.timeMs - dp.energyMapTimeMs);

    printf("\n  Greedy Algorithm:\n");
    printf("    Energy computation: %6.1f%% (%4lld ms)\n", grEnergyPercent, gr.energyMapTimeMs);
    printf("    Seam finding:       %6.1f%% (%4lld ms)\n", grAlgoPercent, gr.timeMs - gr.energyMapTimeMs);

    // Summary and Recommendation
    std::cout << "\n+==================================================================+\n";
    std::cout << "SUMMARY & RECOMMENDATION:\n";
    std::cout << "  DP Advantages:\n";
    std::cout << "    • Guaranteed optimal seams\n";
    std::cout << "    • Lower average seam energy: " << std::fixed << std::setprecision(2)
              << dp.avgSeamEnergy << "\n";
    std::cout << "    • Better visual quality\n\n";

    std::cout << "  Greedy Advantages:\n";
    std::cout << "    • Faster execution (" << std::fixed << std::setprecision(2)
              << speedup << "x speedup)\n";
    std::cout << "    • Lower memory usage\n";
    std::cout << "    • Simple implementation\n\n";

    if (qualityPercent < 10) {
        std::cout << "  RECOMMENDATION: Greedy is acceptable (similar quality, faster)\n";
    } else if (qualityPercent < 20) {
        std::cout << "  RECOMMENDATION: Consider trade-off based on use case\n";
    } else {
        std::cout << "  RECOMMENDATION: Use DP for best quality\n";
    }
    std::cout << "+==================================================================+\n\n";
}

// Function to save stats to file for your report
void saveStatsToFile(const Stats& dp, const Stats& gr, int originalWidth, int originalHeight,
                     int targetWidth, int targetHeight, const std::string& filename) {
    std::string filepath = "../../../" + filename;
    std::ofstream file(filepath, std::ios::app);

    file << "Original Size: " << originalWidth << "x" << originalHeight << " pixels\n";
    file << "Target Size: " << targetWidth << "x" << targetHeight << " pixels\n";
    file << "Vertical Seams Removed: " << (originalWidth - targetWidth) << "\n";
    file << "Horizontal Seams Removed: " << (originalHeight - targetHeight) << "\n";
    file << "Total Seams Removed: " << dp.seamsRemoved << "\n";
    file << "DP Time (ms): " << dp.timeMs << "\n";
    file << "Greedy Time (ms): " << gr.timeMs << "\n";
    file << "Speedup: " << std::fixed << std::setprecision(2)
         << ((float)dp.timeMs / gr.timeMs) << "x\n";
    file << "DP Avg Energy: " << dp.avgSeamEnergy << "\n";
    file << "Greedy Avg Energy: " << gr.avgSeamEnergy << "\n";
    float qualityDiff = ((gr.avgSeamEnergy - dp.avgSeamEnergy) / dp.avgSeamEnergy) * 100.0f;
    file << "Quality Difference: " << qualityDiff << "%\n";
    file << "---\n\n";

    file.close();
    std::cout << "Stats saved to: " << filename << "\n";
}

// =======================================================================
// MAIN
// ======================================================================="

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    std::cout << "       DYNAMIC PROGRAMMING  vs  GREEDY COMPARISON          \n";

    cv::Mat original = cv::imread("../../../Images/tower.jpg");
    if (original.empty()) {
        std::cerr << "Error: Could not load image\n";
        return -1;
    }

    std::cout << "Image loaded: " << original.cols << "x" << original.rows << "\n";
    int targetWidth, targetHeight;
    std::cout << "Enter target width (< " << original.cols << "): ";
    std::cin >> targetWidth;
    std::cout << "Enter target height (< " << original.rows << "): ";
    std::cin >> targetHeight;

    if (targetWidth >= original.cols || targetWidth < 100) {
        std::cerr << "Invalid width.\n";
        return -1;
    }
    if (targetHeight >= original.rows || targetHeight < 100) {
        std::cerr << "Invalid height.\n";
        return -1;
    }

    // --- DP ---
    cv::Mat dpImg = original.clone();
    std::cout << "\nRunning DYNAMIC PROGRAMMING...\n";
    Stats dp = seamCarveWithStats(dpImg, targetWidth, targetHeight, true);

    // --- GREEDY ---
    cv::Mat greedyImg = original.clone();
    std::cout << "\nRunning GREEDY ALGORITHM...\n";
    Stats gr = seamCarveWithStats(greedyImg, targetWidth, targetHeight, false);

    // --- Print detailed results ---
    printDetailedStats(dp, gr, original.cols, original.rows, targetWidth, targetHeight);

    // --- Save results to file for your report ---
    saveStatsToFile(dp, gr, original.cols, original.rows, targetWidth, targetHeight, "benchmark_results.txt");

    // --- Single seam visualization ---
    cv::Mat energy = computeEnergyMap(original);
    auto dpSeam = findVerticalSeamDP(energy);
    auto grSeam = findVerticalSeamGreedy(energy);

    cv::Mat dpVis = visualizeSeam(original, dpSeam, cv::Scalar(0, 255, 0));
    cv::Mat grVis = visualizeSeam(original, grSeam, cv::Scalar(0, 0, 255));

    // Add a separator between seam visualizations
    int sepWidth = 10;
    cv::Mat seamSeparator(dpVis.rows, sepWidth, CV_8UC3, cv::Scalar(200, 200, 200));
    std::vector<cv::Mat> seamParts = { dpVis, seamSeparator, grVis };
    cv::Mat seamComparison;
    cv::hconcat(seamParts, seamComparison);

    cv::putText(seamComparison, "DP (Green)        GREEDY (Red)", { 30, 40 },
        cv::FONT_HERSHEY_SIMPLEX, 1, { 255, 255, 0 }, 2);

    cv::namedWindow("Seam Comparison", cv::WINDOW_NORMAL);
    cv::imshow("Seam Comparison", seamComparison);

    // --- Result Comparison with separator ---
    cv::Mat dpLabelled = dpImg.clone();
    cv::Mat grLabelled = greedyImg.clone();
    cv::putText(dpLabelled, "DP", cv::Point(30, 50),
        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 255), 3);
    cv::putText(grLabelled, "GD", cv::Point(30, 50),
        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 255), 3);

    cv::Mat resultSeparator(dpImg.rows, sepWidth, CV_8UC3, cv::Scalar(200, 200, 200));
    std::vector<cv::Mat> resultParts = { dpLabelled, resultSeparator, grLabelled };
    cv::Mat resultComparison;
    cv::hconcat(resultParts, resultComparison);

    cv::namedWindow("Result Comparison", cv::WINDOW_NORMAL);
    cv::resizeWindow("Result Comparison", 1600, 900);
    cv::imshow("Result Comparison", resultComparison);

    // Save all outputs
    cv::imwrite("../../../dp_result.jpg", dpImg);
    cv::imwrite("../../../greedy_result.jpg", greedyImg);
    cv::imwrite("../../../seam_comparison.jpg", seamComparison);
    cv::imwrite("../../../final_comparison_with_separator.jpg", resultComparison);

    std::cout << "\n Saved: dp_result.jpg, greedy_result.jpg, seam_comparison.jpg, final_comparison_with_separator.jpg\n";
    cv::waitKey(0);
    return 0;

}
