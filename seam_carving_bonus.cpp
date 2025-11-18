#include "seam_carving_bonus.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>

// Forward declarations of core seam carving functions
extern cv::Mat computeEnergyMap(const cv::Mat& image);
extern std::vector<int> findVerticalSeam(const cv::Mat& energy);
extern cv::Mat removeVerticalSeam(const cv::Mat& image, const std::vector<int>& seam);
extern cv::Mat removeHorizontalSeam(const cv::Mat& image);  // No seam parameter version

// ===========================================================================
// VISUALIZATION FUNCTIONS IMPLEMENTATION
// ===========================================================================

cv::Mat visualizeEnergyMap(const cv::Mat& energy) {
    cv::Mat normalized, colorMap;
    cv::normalize(energy, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(normalized, colorMap, cv::COLORMAP_JET);
    return colorMap;
}

cv::Mat drawSeam(const cv::Mat& image, const std::vector<int>& seam,
    cv::Scalar color, int thickness) {
    cv::Mat result = image.clone();
    for (int i = 0; i < (int)seam.size(); i++) {
        cv::circle(result, cv::Point(seam[i], i), thickness / 2, color, -1);
        if (i > 0) {
            cv::line(result, cv::Point(seam[i - 1], i - 1),
                cv::Point(seam[i], i), color, thickness);
        }
    }
    return result;
}

cv::Mat drawMultipleSeams(const cv::Mat& image, const std::deque<std::vector<int>>& seams) {
    cv::Mat result = image.clone();

    // Color gradient from red (oldest) to bright green (newest)
    for (size_t idx = 0; idx < seams.size(); idx++) {
        float ratio = (float)idx / std::max(1.0f, (float)(seams.size() - 1));
        cv::Scalar color(0, 255 * ratio, 255 * (1 - ratio));  // BGR: from red to green

        // Adjust seam positions to account for previously removed seams
        std::vector<int> adjustedSeam = seams[idx];
        for (size_t prev = 0; prev < idx; prev++) {
            for (int i = 0; i < (int)adjustedSeam.size(); i++) {
                if (seams[prev][i] <= adjustedSeam[i]) {
                    adjustedSeam[i]++;
                }
            }
        }

        // Draw seam as connected line with circles at each point
        for (int i = 0; i < (int)adjustedSeam.size(); i++) {
            int col = adjustedSeam[i];

            if (col >= 0 && col < result.cols) {
                // Draw circle at this point
                cv::circle(result, cv::Point(col, i), 1, color, -1);

                // Draw line connecting to previous point
                if (i > 0) {
                    int prevCol = adjustedSeam[i - 1];
                    if (prevCol >= 0 && prevCol < result.cols) {
                        cv::line(result, cv::Point(prevCol, i - 1),
                            cv::Point(col, i), color, 1);
                    }
                }
            }
        }
    }
    return result;
}

cv::Mat createMultiPanelView(const cv::Mat& original, const cv::Mat& current,
    const cv::Mat& energy, const cv::Mat& seamVis) {
    int panelHeight = 400;
    float scale;
    cv::Mat scaled_original, scaled_current, scaled_energy, scaled_seam;

    // Scale all panels to same height
    scale = (float)panelHeight / original.rows;
    cv::resize(original, scaled_original, cv::Size((int)(original.cols * scale), panelHeight));

    scale = (float)panelHeight / current.rows;
    cv::resize(current, scaled_current, cv::Size((int)(current.cols * scale), panelHeight));

    cv::Mat energyVis = visualizeEnergyMap(energy);
    scale = (float)panelHeight / energyVis.rows;
    cv::resize(energyVis, scaled_energy, cv::Size((int)(energyVis.cols * scale), panelHeight));

    scale = (float)panelHeight / seamVis.rows;
    cv::resize(seamVis, scaled_seam, cv::Size((int)(seamVis.cols * scale), panelHeight));

    // Create 2x2 grid
    int maxWidth = std::max(scaled_original.cols + scaled_current.cols,
        scaled_energy.cols + scaled_seam.cols) + 30;
    cv::Mat multiPanel(panelHeight * 2 + 60, maxWidth, CV_8UC3, cv::Scalar(30, 30, 30));

    // Place panels
    scaled_original.copyTo(multiPanel(cv::Rect(0, 30, scaled_original.cols, panelHeight)));
    scaled_current.copyTo(multiPanel(cv::Rect(scaled_original.cols + 20, 30,
        scaled_current.cols, panelHeight)));
    scaled_energy.copyTo(multiPanel(cv::Rect(0, panelHeight + 40,
        scaled_energy.cols, panelHeight)));
    scaled_seam.copyTo(multiPanel(cv::Rect(scaled_energy.cols + 20, panelHeight + 40,
        scaled_seam.cols, panelHeight)));

    // Add labels
    cv::putText(multiPanel, "ORIGINAL", cv::Point(10, 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(multiPanel, "CURRENT", cv::Point(scaled_original.cols + 20, 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(multiPanel, "ENERGY MAP", cv::Point(10, panelHeight + 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(multiPanel, "SEAM VISUALIZATION", cv::Point(scaled_energy.cols + 20, panelHeight + 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    return multiPanel;
}

// ===========================================================================
// INTERACTIVE SEAM CARVING WITH VISUALIZATION
// ===========================================================================

cv::Mat seamCarveWithVisualization(cv::Mat image, int targetWidth, int targetHeight,
    const std::string& windowName,
    VisualizationSettings& settings) {
    std::cout << "\n[START] Starting INTERACTIVE seam carving with visualization...\n";
    std::cout << "--------------------------------------\n";
    std::cout << "From: " << image.cols << "x" << image.rows
        << " → To: " << targetWidth << "x" << targetHeight << "\n\n";

    std::cout << "CONTROLS DURING VISUALIZATION:\n";
    std::cout << "  [SPACE] - Pause/Resume\n";
    std::cout << "  [S]     - Toggle seam display\n";
    std::cout << "  [E]     - Show energy map\n";
    std::cout << "  [+/-]   - Speed up/slow down\n";
    std::cout << "  [Q]     - Quit visualization\n";
    std::cout << "--------------------------------------\n\n";

    int totalVerticalSeams = image.cols - targetWidth;
    int totalHorizontalSeams = image.rows - targetHeight;
    int seamCount = 0;
    int totalSeams = totalVerticalSeams + totalHorizontalSeams;

    settings.seamHistory.clear();
    bool showEnergyMap = false;

    // Remove vertical seams with visualization
    for (int i = 0; i < totalVerticalSeams; i++) {
        seamCount++;
        cv::Mat energy = computeEnergyMap(image);
        std::vector<int> seam = findVerticalSeam(energy);

        if (settings.visualizeSeams) {
            // Store seam for history
            if (settings.seamHistory.size() >= (size_t)settings.maxSeamsToShow) {
                settings.seamHistory.pop_front();
            }
            settings.seamHistory.push_back(seam);

            // Create visualization
            cv::Mat display = image.clone();
            cv::Mat visualization;

            if (showEnergyMap) {
                cv::Mat energyVis = visualizeEnergyMap(energy);
                display = drawSeam(display, seam, cv::Scalar(0, 255, 255), 2);
                cv::hconcat(display, energyVis, visualization);
            }
            else {
                // Draw current seam in yellow with thicker line
                display = drawSeam(display, seam, cv::Scalar(0, 255, 255), 2);
                // Draw history seams
                display = drawMultipleSeams(display, settings.seamHistory);
            }

            // Add info overlay
            cv::putText(display, "VERTICAL SEAM " + std::to_string(i + 1) + "/" +
                std::to_string(totalVerticalSeams),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);
            cv::putText(display, "Progress: " + std::to_string(seamCount) + "/" +
                std::to_string(totalSeams),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);
            cv::putText(display, "Speed: " + std::to_string(1000 / settings.visualizationSpeed) +
                " seams/sec",
                cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);

            cv::imshow(windowName, showEnergyMap ? visualization : display);

            // Handle keyboard input
            int key = cv::waitKey(settings.visualizationSpeed);
            if (key == ' ') {
                settings.pauseVisualization = !settings.pauseVisualization;
                while (settings.pauseVisualization) {
                    key = cv::waitKey(100);
                    if (key == ' ') break;
                    if (key == 'q' || key == 'Q') return image;
                }
            }
            else if (key == 's' || key == 'S') {
                settings.visualizeSeams = !settings.visualizeSeams;
            }
            else if (key == 'e' || key == 'E') {
                showEnergyMap = !showEnergyMap;
            }
            else if (key == '+' || key == '=') {
                settings.visualizationSpeed = std::max(10, settings.visualizationSpeed - 20);
            }
            else if (key == '-') {
                settings.visualizationSpeed = std::min(500, settings.visualizationSpeed + 20);
            }
            else if (key == 'q' || key == 'Q') {
                std::cout << "\n[WARNING] Visualization stopped by user\n";
                return image;
            }
        }

        image = removeVerticalSeam(image, seam);

        if ((i + 1) % 10 == 0 || i == 0) {
            std::cout << "  [OK] Vertical seam " << (i + 1) << "/" << totalVerticalSeams
                << " (Size: " << image.cols << "x" << image.rows << ")\n";
        }
    }

    settings.seamHistory.clear();

    // Remove horizontal seams with visualization
    for (int i = 0; i < totalHorizontalSeams; i++) {
        seamCount++;

        if (settings.visualizeSeams) {
            cv::Mat display = image.clone();

            // Add info overlay (can't visualize horizontal seam with current implementation)
            cv::putText(display, "HORIZONTAL SEAM " + std::to_string(i + 1) + "/" +
                std::to_string(totalHorizontalSeams),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
            cv::putText(display, "Progress: " + std::to_string(seamCount) + "/" +
                std::to_string(totalSeams),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);

            cv::imshow(windowName, display);

            int key = cv::waitKey(settings.visualizationSpeed);
            if (key == ' ') {
                settings.pauseVisualization = !settings.pauseVisualization;
                while (settings.pauseVisualization) {
                    key = cv::waitKey(100);
                    if (key == ' ') break;
                    if (key == 'q' || key == 'Q') return image;
                }
            }
            else if (key == 'q' || key == 'Q') {
                std::cout << "\n[WARNING] Visualization stopped by user\n";
                return image;
            }
        }

        image = removeHorizontalSeam(image);

        if ((i + 1) % 10 == 0 || i == 0) {
            std::cout << "  [OK] Horizontal seam " << (i + 1) << "/" << totalHorizontalSeams
                << " (Size: " << image.cols << "x" << image.rows << ")\n";
        }
    }

    std::cout << "\n[DONE] Seam carving complete! Final size: " << image.cols << "x"
        << image.rows << "\n";
    return image;
}

// ===========================================================================
// INTERACTIVE MOUSE SELECTION
// ===========================================================================

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    MouseData* data = (MouseData*)userdata;

    if (event == cv::EVENT_LBUTTONDOWN) {
        data->startPoint = cv::Point(x, y);
        data->endPoint = cv::Point(x, y);
        data->selecting = true;
        data->selectionDone = false;
    }
    else if (event == cv::EVENT_MOUSEMOVE) {
        if (data->selecting) {
            data->endPoint = cv::Point(x, y);
        }
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        data->endPoint = cv::Point(x, y);
        data->selecting = false;
        data->selectionDone = true;

        // Calculate selection rectangle
        int x1 = std::min(data->startPoint.x, data->endPoint.x);
        int y1 = std::min(data->startPoint.y, data->endPoint.y);
        int x2 = std::max(data->startPoint.x, data->endPoint.x);
        int y2 = std::max(data->startPoint.y, data->endPoint.y);

        data->selection = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }
}

cv::Rect interactiveMouseSelection(const cv::Mat& image, const std::string& windowName) {
    std::cout << "\n[MOUSE]  INTERACTIVE MOUSE SELECTION MODE\n";
    std::cout << "Click and drag to select target area\n";
    std::cout << "Press ESC to cancel\n\n";

    MouseData mouseData;
    mouseData.image = image.clone();

    cv::setMouseCallback(windowName, mouseCallback, &mouseData);

    while (!mouseData.selectionDone) {
        cv::Mat display = mouseData.image.clone();

        if (mouseData.selecting) {
            cv::rectangle(display, mouseData.startPoint, mouseData.endPoint,
                cv::Scalar(0, 255, 0), 2);

            int targetW = abs(mouseData.endPoint.x - mouseData.startPoint.x);
            int targetH = abs(mouseData.endPoint.y - mouseData.startPoint.y);
            std::string text = "Target: " + std::to_string(targetW) + "x" + std::to_string(targetH);
            cv::putText(display, text, cv::Point(mouseData.startPoint.x, mouseData.startPoint.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow(windowName, display);
        if (cv::waitKey(30) == 27) {  // ESC to cancel
            mouseData.selection = cv::Rect(0, 0, 0, 0);
            break;
        }
    }

    cv::setMouseCallback(windowName, nullptr, nullptr);
    return mouseData.selection;
}

// ===========================================================================
// STEP-BY-STEP MODE
// ===========================================================================

void runStepByStepMode(cv::Mat& image, const std::string& windowName) {
    std::cout << "\n[STEP] STEP-BY-STEP MODE\n";
    std::cout << "[V] - Remove vertical seam\n";
    std::cout << "[H] - Remove horizontal seam\n";
    std::cout << "[ESC] - Exit step mode\n\n";

    bool stepMode = true;
    while (stepMode && image.cols > 100 && image.rows > 100) {
        cv::Mat energy = computeEnergyMap(image);
        cv::Mat display = image.clone();

        // Show current state
        cv::putText(display, "Step Mode - Press V/H to remove seam, ESC to exit",
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
            cv::Scalar(0, 255, 255), 2);
        cv::putText(display, "Size: " + std::to_string(image.cols) + "x" +
            std::to_string(image.rows),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
            cv::Scalar(255, 255, 255), 2);

        cv::imshow(windowName, display);

        int key = cv::waitKey(0);
        if (key == 27) {  // ESC
            stepMode = false;
        }
        else if (key == 'v' || key == 'V') {
            std::vector<int> seam = findVerticalSeam(energy);
            display = drawSeam(image, seam, cv::Scalar(0, 0, 255), 3);
            cv::imshow(windowName, display);
            cv::waitKey(500);
            image = removeVerticalSeam(image, seam);
            std::cout << "Removed vertical seam. New size: " << image.cols << "x"
                << image.rows << "\n";
        }
        else if (key == 'h' || key == 'H') {
            // Show a brief indicator before removing
            cv::putText(display, "Removing horizontal seam...",
                cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
            cv::imshow(windowName, display);
            cv::waitKey(300);
            image = removeHorizontalSeam(image);
            std::cout << "Removed horizontal seam. New size: " << image.cols << "x"
                << image.rows << "\n";
        }
    }
}

// ===========================================================================
// BATCH RESIZE PRESETS
// ===========================================================================

bool calculatePresetDimensions(int currentWidth, int currentHeight, int preset,
    int& targetWidth, int& targetHeight) {
    targetWidth = currentWidth;
    targetHeight = currentHeight;

    switch (preset) {
    case 1:  // 16:9 aspect ratio
        targetHeight = (currentWidth * 9) / 16;
        break;
    case 2:  // 4:3 aspect ratio
        targetHeight = (currentWidth * 3) / 4;
        break;
    case 3:  // 1:1 square
        targetWidth = targetHeight = std::min(currentWidth, currentHeight);
        break;
    case 4:  // 50% reduction
        targetWidth = currentWidth / 2;
        targetHeight = currentHeight / 2;
        break;
    default:
        return false;
    }

    targetWidth = std::max(100, targetWidth);
    targetHeight = std::max(100, targetHeight);

    return true;
}

// ===========================================================================
// VISUALIZATION SETTINGS MENU
// ===========================================================================

void handleVisualizationSettings(VisualizationSettings& settings) {
    std::cout << "\n[SETTINGS]  VISUALIZATION SETTINGS\n";
    std::cout << "Current speed: " << settings.visualizationSpeed << "ms per seam\n";
    std::cout << "Show seams: " << (settings.visualizeSeams ? "ON" : "OFF") << "\n";
    std::cout << "Max seams in history: " << settings.maxSeamsToShow << "\n\n";

    std::cout << "[1] Change speed (10-500ms)\n";
    std::cout << "[2] Toggle seam visualization\n";
    std::cout << "[3] Change history size (1-50)\n";
    std::cout << "Choice: ";

    int settingChoice;
    std::cin >> settingChoice;

    switch (settingChoice) {
    case 1:
        std::cout << "Enter speed (ms): ";
        std::cin >> settings.visualizationSpeed;
        settings.visualizationSpeed = std::max(10, std::min(500, settings.visualizationSpeed));
        break;
    case 2:
        settings.visualizeSeams = !settings.visualizeSeams;
        std::cout << "Seam visualization: " << (settings.visualizeSeams ? "ON" : "OFF") << "\n";
        break;
    case 3:
        std::cout << "Enter history size: ";
        std::cin >> settings.maxSeamsToShow;
        settings.maxSeamsToShow = std::max(1, std::min(50, settings.maxSeamsToShow));
        break;
    }
}

// ===========================================================================
// SAVE WITH VISUALIZATION EXAMPLES
// ===========================================================================

bool saveWithVisualizationExamples(const cv::Mat& image, const std::string& baseFilename) {
    std::string mainFilename = baseFilename + "_" + std::to_string(image.cols) + "x" +
        std::to_string(image.rows) + ".jpg";

    if (!cv::imwrite(mainFilename, image)) {
        return false;
    }

    std::cout << "[OK] Saved main image to: " << mainFilename << "\n";

    // Save energy map visualization
    cv::Mat energy = computeEnergyMap(image);
    cv::Mat energyVis = visualizeEnergyMap(energy);
    std::string energyFilename = baseFilename + "_energy_map.jpg";
    cv::imwrite(energyFilename, energyVis);
    std::cout << "[OK] Saved energy map to: " << energyFilename << "\n";

    // Save seam visualization
    std::vector<int> seam = findVerticalSeam(energy);
    cv::Mat seamVis = drawSeam(image, seam, cv::Scalar(0, 0, 255), 3);
    std::string seamFilename = baseFilename + "_seam_visualization.jpg";
    cv::imwrite(seamFilename, seamVis);
    std::cout << "[OK] Saved seam visualization to: " << seamFilename << "\n";

    return true;
}