#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <chrono>

// ═══════════════════════════════════════════════════════════════════════════
// ENERGY MAP COMPUTATION (Shared by DP & Greedy)
// ═══════════════════════════════════════════════════════════════════════════

cv::Mat computeEnergyMap(const cv::Mat & image) {
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

    // Per-channel Scharr gradients + small Laplacian texture boost
    std::vector<cv::Mat> ch; cv::split(img32f, ch);
    cv::Mat energy = cv::Mat::zeros(image.size(), CV_32F);
    for (int c = 0; c < 3; ++c) {
        cv::Mat gx, gy;
        cv::Scharr(ch[c], gx, CV_32F, 1, 0);
        cv::Scharr(ch[c], gy, CV_32F, 0, 1);
        energy += gx.mul(gx) + gy.mul(gy);
    }
    cv::sqrt(energy, energy);

    // Add Laplacian component to protect fine details
    cv::Mat gray, lap, absLap;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_32F, 3);
    cv::absdiff(lap, 0, absLap);
    energy += 0.2f * absLap;

    cv::GaussianBlur(energy, energy, cv::Size(3, 3), 0);
    return energy;
}

// ═══════════════════════════════════════════════════════════════════════════
// DYNAMIC PROGRAMMING SEAM FINDING
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// GREEDY SEAM FINDING (Improved Lookahead Version)
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> findVerticalSeamGreedy(const cv::Mat& energy, int lookaheadDepth = 3) {
    const int rows = energy.rows, cols = energy.cols;
    std::vector<int> seam(rows);

    int currentCol = 0;
    float best = energy.at<float>(0, 0);
    for (int j = 1; j < cols; ++j) {
        float v = energy.at<float>(0, j);
        if (v < best) { best = v; currentCol = j; }
    }
    seam[0] = currentCol;

    auto estimatePathEnergy = [&](int startRow, int startCol, int depth) {
        float total = 0.f;
        int col = startCol;
        for (int d = 0; d < depth && (startRow + d) < rows; ++d) {
            float bestE = FLT_MAX;
            int bestC = col;
            for (int k = col - 1; k <= col + 1; ++k) {
                if (k >= 0 && k < cols) {
                    float e = energy.at<float>(startRow + d, k);
                    if (e < bestE) { bestE = e; bestC = k; }
                }
            }
            total += bestE;
            col = bestC;
        }
        return total;
        };

    for (int i = 1; i < rows; ++i) {
        int bestCol = currentCol;
        float bestFuture = FLT_MAX;
        for (int k = currentCol - 1; k <= currentCol + 1; ++k) {
            if (k < 0 || k >= cols) continue;
            float future = estimatePathEnergy(i, k, lookaheadDepth);
            if (future < bestFuture) { bestFuture = future; bestCol = k; }
        }
        seam[i] = bestCol;
        currentCol = bestCol;
    }
    return seam;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMMON UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// SEAM CARVING COMPARISON LOGIC
// ═══════════════════════════════════════════════════════════════════════════

struct Stats {
    long long timeMs;
    float totalEnergy;
    int seamsRemoved;
};

Stats seamCarveWithStats(cv::Mat& img, int targetWidth, bool useDP) {
    Stats s{ 0, 0.f, 0 };
    int seams = img.cols - targetWidth;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < seams; i++) {
        cv::Mat energy = computeEnergyMap(img);
        std::vector<int> seam = useDP ?
            findVerticalSeamDP(energy) : findVerticalSeamGreedy(energy, 5);

        s.totalEnergy += calculateSeamEnergy(energy, seam);
        s.seamsRemoved++;
        img = removeVerticalSeam(img, seam);
    }

    s.timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start)
        .count();
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║       DYNAMIC PROGRAMMING  vs  GREEDY COMPARISON          ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    cv::Mat original = cv::imread("../../../Images/ghibli.jpg");
    if (original.empty()) {
        std::cerr << "✗ Error: Could not load image\n";
        return -1;
    }

    std::cout << "✓ Image loaded: " << original.cols << "x" << original.rows << "\n";
    int targetWidth;
    std::cout << "Enter target width (< " << original.cols << "): ";
    std::cin >> targetWidth;
    if (targetWidth >= original.cols || targetWidth < 100) {
        std::cerr << "Invalid width.\n";
        return -1;
    }

    // --- DP ---
    cv::Mat dpImg = original.clone();
    std::cout << "\n▶ Running DYNAMIC PROGRAMMING...\n";
    Stats dp = seamCarveWithStats(dpImg, targetWidth, true);

    // --- GREEDY ---
    cv::Mat greedyImg = original.clone();
    std::cout << "\n▶ Running GREEDY (lookahead=5)...\n";
    Stats gr = seamCarveWithStats(greedyImg, targetWidth, false);

    // --- Results ---
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                        RESULTS                            ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    printf("║ Metric             │ DP             │ Greedy (lookahead)  ║\n");
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    printf("║ Time (ms)          │ %-13lld │ %-18lld ║\n", dp.timeMs, gr.timeMs);
    float dpAvg = dp.totalEnergy / dp.seamsRemoved;
    float grAvg = gr.totalEnergy / gr.seamsRemoved;
    printf("║ Avg Seam Energy    │ %-13.2f │ %-18.2f ║\n", dpAvg, grAvg);
    printf("║ Total Energy       │ %-13.2f │ %-18.2f ║\n", dp.totalEnergy, gr.totalEnergy);
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    // --- Single seam visualization ---
    cv::Mat energy = computeEnergyMap(original);
    auto dpSeam = findVerticalSeamDP(energy);
    auto grSeam = findVerticalSeamGreedy(energy, 5);

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
    cv::imwrite("dp_result.jpg", dpImg);
    cv::imwrite("greedy_result.jpg", greedyImg);
    cv::imwrite("seam_comparison.jpg", seamComparison);
    cv::imwrite("final_comparison_with_separator.jpg", resultComparison);

    std::cout << "\n✓ Saved: dp_result.jpg, greedy_result.jpg, seam_comparison.jpg, final_comparison_with_separator.jpg\n";
    cv::waitKey(0);
    return 0;

}
