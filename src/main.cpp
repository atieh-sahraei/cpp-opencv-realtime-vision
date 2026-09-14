#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string formatValue(double value, int precision = 2)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

void drawFlowArrows(
    cv::Mat& frame,
    const cv::Mat& flow,
    int step = 24,
    float minMagnitude = 0.8F)
{
    for (int y = step / 2; y < flow.rows; y += step) {
        for (int x = step / 2; x < flow.cols; x += step) {
            const cv::Point2f vector = flow.at<cv::Point2f>(y, x);
            const float magnitude = cv::norm(vector);

            if (magnitude < minMagnitude) {
                continue;
            }

            const cv::Point start(x, y);
            const cv::Point end(
                cvRound(x + vector.x),
                cvRound(y + vector.y)
            );

            cv::arrowedLine(
                frame,
                start,
                end,
                cv::Scalar(255, 120, 0),
                1,
                cv::LINE_AA,
                0,
                0.25
            );
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::string inputPath;
    if (argc > 1) {
        inputPath = argv[1];
    }

    cv::VideoCapture cap;

    if (inputPath.empty()) {
        cap.open(0);
        std::cout << "Opening default camera..." << std::endl;
    } else {
        cap.open(inputPath);
        std::cout << "Opening video: " << inputPath << std::endl;
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera or video." << std::endl;
        return 1;
    }

    cv::Mat frame;
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat binary;
    cv::Mat previousGray;
    cv::Mat flow;

    const double minContourArea = 500.0;

    // Exponential moving average for temporal smoothing.
    // Smaller alpha = smoother but slower response.
    const double smoothingAlpha = 0.15;
    double smoothedMotion = 0.0;
    bool smoothingInitialized = false;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        const auto frameStart = std::chrono::steady_clock::now();

        // ------------------------------------------------------------
        // 1. Image preprocessing
        // ------------------------------------------------------------
        const auto preprocessStart = std::chrono::steady_clock::now();

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        const auto preprocessEnd = std::chrono::steady_clock::now();

        // ------------------------------------------------------------
        // 2. Segmentation and geometric measurement
        // ------------------------------------------------------------
        const auto segmentationStart = std::chrono::steady_clock::now();

        cv::threshold(
            blurred,
            binary,
            0,
            255,
            cv::THRESH_BINARY + cv::THRESH_OTSU
        );

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            binary,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        double largestArea = 0.0;
        int largestIndex = -1;

        for (size_t i = 0; i < contours.size(); ++i) {
            const double area = cv::contourArea(contours[i]);

            if (area > minContourArea && area > largestArea) {
                largestArea = area;
                largestIndex = static_cast<int>(i);
            }
        }

        if (largestIndex >= 0) {
            const cv::RotatedRect box = cv::minAreaRect(contours[largestIndex]);

            cv::Point2f vertices[4];
            box.points(vertices);

            for (int i = 0; i < 4; ++i) {
                cv::line(
                    frame,
                    vertices[i],
                    vertices[(i + 1) % 4],
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA
                );
            }

            const float objectWidth = std::min(box.size.width, box.size.height);
            const float objectLength = std::max(box.size.width, box.size.height);

            cv::putText(
                frame,
                "Length: " + formatValue(objectLength, 1) + " px",
                cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(0, 255, 0),
                2
            );

            cv::putText(
                frame,
                "Width: " + formatValue(objectWidth, 1) + " px",
                cv::Point(20, 62),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(0, 255, 0),
                2
            );
        }

        const auto segmentationEnd = std::chrono::steady_clock::now();

        // ------------------------------------------------------------
        // 3. Dense optical flow and temporal filtering
        // ------------------------------------------------------------
        const auto flowStart = std::chrono::steady_clock::now();

        double meanMotion = 0.0;

        if (!previousGray.empty()) {
            cv::calcOpticalFlowFarneback(
                previousGray,
                gray,
                flow,
                0.5,  // pyramid scale
                3,    // pyramid levels
                15,   // window size
                3,    // iterations per pyramid level
                5,    // polynomial neighbourhood size
                1.2,  // polynomial sigma
                0
            );

            std::vector<cv::Mat> flowChannels(2);
            cv::split(flow, flowChannels);

            cv::Mat magnitude;
            cv::Mat angle;
            cv::cartToPolar(
                flowChannels[0],
                flowChannels[1],
                magnitude,
                angle,
                false
            );

            meanMotion = cv::mean(magnitude)[0];

            if (!smoothingInitialized) {
                smoothedMotion = meanMotion;
                smoothingInitialized = true;
            } else {
                smoothedMotion =
                    smoothingAlpha * meanMotion +
                    (1.0 - smoothingAlpha) * smoothedMotion;
            }

            drawFlowArrows(frame, flow);
        }

        gray.copyTo(previousGray);

        const auto flowEnd = std::chrono::steady_clock::now();
        const auto frameEnd = std::chrono::steady_clock::now();

        // ------------------------------------------------------------
        // 4. Runtime profiling
        // ------------------------------------------------------------
        const double preprocessMs =
            std::chrono::duration<double, std::milli>(
                preprocessEnd - preprocessStart
            ).count();

        const double segmentationMs =
            std::chrono::duration<double, std::milli>(
                segmentationEnd - segmentationStart
            ).count();

        const double flowMs =
            std::chrono::duration<double, std::milli>(
                flowEnd - flowStart
            ).count();

        const double totalMs =
            std::chrono::duration<double, std::milli>(
                frameEnd - frameStart
            ).count();

        const double processingFps = totalMs > 0.0 ? 1000.0 / totalMs : 0.0;

        cv::putText(
            frame,
            "Motion raw: " + formatValue(meanMotion) + " px/frame",
            cv::Point(20, 95),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(255, 120, 0),
            2
        );

        cv::putText(
            frame,
            "Motion smooth: " + formatValue(smoothedMotion) + " px/frame",
            cv::Point(20, 122),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(255, 120, 0),
            2
        );

        cv::putText(
            frame,
            "Processing FPS: " + formatValue(processingFps, 1),
            cv::Point(20, 149),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 255),
            2
        );

        cv::putText(
            frame,
            "Time [ms] pre:" + formatValue(preprocessMs, 1) +
                " seg:" + formatValue(segmentationMs, 1) +
                " flow:" + formatValue(flowMs, 1) +
                " total:" + formatValue(totalMs, 1),
            cv::Point(20, 176),
            cv::FONT_HERSHEY_SIMPLEX,
            0.52,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA
        );

        cv::imshow("C++ OpenCV Vision Pipeline", frame);
        cv::imshow("Binary Segmentation", binary);

        const char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
