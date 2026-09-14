#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>

int main(int argc, char** argv)
{
    std::string inputPath;
    if (argc > 1) inputPath = argv[1];

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

    cv::Mat frame, gray, blurred, binary;
    const double minContourArea = 500.0;

    auto lastTime = std::chrono::high_resolution_clock::now();
    double fps = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

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
            double area = cv::contourArea(contours[i]);
            if (area > minContourArea && area > largestArea) {
                largestArea = area;
                largestIndex = static_cast<int>(i);
            }
        }

        if (largestIndex >= 0) {
            cv::RotatedRect box = cv::minAreaRect(contours[largestIndex]);

            cv::Point2f vertices[4];
            box.points(vertices);

            for (int i = 0; i < 4; ++i) {
                cv::line(
                    frame,
                    vertices[i],
                    vertices[(i + 1) % 4],
                    cv::Scalar(0, 255, 0),
                    2
                );
            }

            float objectWidth = std::min(box.size.width, box.size.height);
            float objectLength = std::max(box.size.width, box.size.height);

            cv::putText(
                frame,
                "Length: " + std::to_string(static_cast<int>(objectLength)) + " px",
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 0),
                2
            );

            cv::putText(
                frame,
                "Width: " + std::to_string(static_cast<int>(objectWidth)) + " px",
                cv::Point(20, 70),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 0),
                2
            );
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - lastTime;
        lastTime = currentTime;

        if (elapsed.count() > 0.0) {
            fps = 1.0 / elapsed.count();
        }

        cv::putText(
            frame,
            "FPS: " + std::to_string(static_cast<int>(fps)),
            cv::Point(20, 100),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 255),
            2
        );

        cv::imshow("C++ OpenCV Vision Pipeline", frame);
        cv::imshow("Binary Segmentation", binary);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
