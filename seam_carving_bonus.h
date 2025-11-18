#ifndef SEAM_CARVING_BONUS_H
#define SEAM_CARVING_BONUS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════
// BONUS FEATURES: VISUALIZATION AND INTERACTIVE TOOLS
// ═══════════════════════════════════════════════════════════════════════════

// Global visualization settings
struct VisualizationSettings {
    bool visualizeSeams = true;
    int visualizationSpeed = 50;  // milliseconds between frames
    bool pauseVisualization = false;
    int maxSeamsToShow = 10;
    std::deque<std::vector<int>> seamHistory;
};

// Mouse callback data structure for interactive region selection
struct MouseData {
    cv::Mat image;
    cv::Point startPoint;
    cv::Point endPoint;
    bool selecting = false;
    bool selectionDone = false;
    cv::Rect selection;
};

// ═══════════════════════════════════════════════════════════════════════════
// VISUALIZATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Visualize energy map as a heatmap using colormap
 * @param energy The energy matrix (CV_32F)
 * @return Colored visualization of the energy map
 */
cv::Mat visualizeEnergyMap(const cv::Mat& energy);

/**
 * @brief Draw a single seam on the image
 * @param image Source image
 * @param seam Vector of column indices (for vertical seam) or row indices (for horizontal)
 * @param color Color of the seam (default: red)
 * @param thickness Line thickness (default: 2)
 * @return Image with seam drawn
 */
cv::Mat drawSeam(const cv::Mat& image, const std::vector<int>& seam,
    cv::Scalar color = cv::Scalar(0, 0, 255), int thickness = 2);

/**
 * @brief Draw multiple seams with color gradient (red to green)
 * @param image Source image
 * @param seams Deque of seams to draw
 * @return Image with multiple seams drawn
 */
cv::Mat drawMultipleSeams(const cv::Mat& image, const std::deque<std::vector<int>>& seams);

/**
 * @brief Create multi-panel view showing original, current, energy map, and seam visualization
 * @param original Original image
 * @param current Current processed image
 * @param energy Energy map
 * @param seamVis Seam visualization
 * @return Combined multi-panel image
 */
cv::Mat createMultiPanelView(const cv::Mat& original, const cv::Mat& current,
    const cv::Mat& energy, const cv::Mat& seamVis);

// ═══════════════════════════════════════════════════════════════════════════
// INTERACTIVE SEAM CARVING WITH VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Perform seam carving with real-time visualization
 * @param image Source image
 * @param targetWidth Target width
 * @param targetHeight Target height
 * @param windowName Window name for display
 * @param settings Visualization settings
 * @return Resized image
 *
 * Controls during visualization:
 * - SPACE: Pause/Resume
 * - S: Toggle seam display
 * - E: Show energy map
 * - +/-: Speed up/slow down
 * - Q: Quit visualization
 */
cv::Mat seamCarveWithVisualization(cv::Mat image, int targetWidth, int targetHeight,
    const std::string& windowName,
    VisualizationSettings& settings);

// ═══════════════════════════════════════════════════════════════════════════
// INTERACTIVE MOUSE SELECTION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Mouse callback function for interactive region selection
 * @param event OpenCV mouse event
 * @param x Mouse x coordinate
 * @param y Mouse y coordinate
 * @param flags Mouse event flags
 * @param userdata Pointer to MouseData structure
 */
void mouseCallback(int event, int x, int y, int flags, void* userdata);

/**
 * @brief Interactive mouse-based region selection for target dimensions
 * @param image Source image to display
 * @param windowName Window name for display
 * @return Rectangle representing the selected region (width and height)
 */
cv::Rect interactiveMouseSelection(const cv::Mat& image, const std::string& windowName);

// ═══════════════════════════════════════════════════════════════════════════
// STEP-BY-STEP MODE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Run step-by-step interactive mode for seam removal
 * @param image Reference to current image (will be modified)
 * @param windowName Window name for display
 *
 * Controls:
 * - V: Remove vertical seam
 * - H: Remove horizontal seam
 * - ESC: Exit step mode
 */
void runStepByStepMode(cv::Mat& image, const std::string& windowName);

// ═══════════════════════════════════════════════════════════════════════════
// BATCH RESIZE PRESETS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Calculate target dimensions based on preset aspect ratios
 * @param currentWidth Current image width
 * @param currentHeight Current image height
 * @param preset Preset number (1: 16:9, 2: 4:3, 3: 1:1 square, 4: 50% reduction)
 * @param targetWidth Output target width
 * @param targetHeight Output target height
 * @return true if preset is valid, false otherwise
 */
bool calculatePresetDimensions(int currentWidth, int currentHeight, int preset,
    int& targetWidth, int& targetHeight);

// ═══════════════════════════════════════════════════════════════════════════
// VISUALIZATION SETTINGS MENU
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Display and handle visualization settings menu
 * @param settings Reference to visualization settings to modify
 */
void handleVisualizationSettings(VisualizationSettings& settings);

// ═══════════════════════════════════════════════════════════════════════════
// SAVE WITH VISUALIZATION EXAMPLES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Save current image along with visualization examples
 * @param image Current image to save
 * @param baseFilename Base filename (without extension)
 * @return true if successful, false otherwise
 */
bool saveWithVisualizationExamples(const cv::Mat& image, const std::string& baseFilename);

#endif // SEAM_CARVING_BONUS_H