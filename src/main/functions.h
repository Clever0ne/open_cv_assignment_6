#pragma once

#include <cstdint>

#include "opencv2/highgui.hpp"

enum class Stage
{
    STAGE_ONE = 0,
    STAGE_TWO = 1
};

bool detectStructuredLight(cv::Mat &inputImage, cv::Mat &outputImage);
bool drawMap(cv::Mat &inputImage, cv::Mat &outputImage);

void waitForKeyPress(const uint32_t delay);
char waitForAnswer(const cv::String &question);

inline int32_t cvtHue(const int32_t hue)
{
    return static_cast<int32_t>(static_cast<float>(hue) / 360.0f * 179.0f);
}

inline int32_t cvtSat(const int32_t sat)
{
    return static_cast<int32_t>(static_cast<float>(sat) / 100.0f * 255.0f);
}

inline int32_t cvtVal(const int32_t val)
{
    return static_cast<int32_t>(static_cast<float>(val) / 100.0f * 255.0f);
}

void getLowerAndUpperHSVBounds(const int32_t hueLow, const int32_t hueHigh,
                               const int32_t satLow, const int32_t satHigh,
                               const int32_t valLow, const int32_t valHigh,
                               std::vector<cv::Scalar> &lowerBound,
                               std::vector<cv::Scalar> &upperBound);

bool thinnig(cv::Mat &inputImage, cv::Mat &outputImage);

uint32_t countWhitePixels(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2);
uint32_t countTransitions(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2);
bool areBorderPixelsBlack(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2, Stage stage);
