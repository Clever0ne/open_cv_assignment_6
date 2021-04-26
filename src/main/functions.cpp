#include <iostream>

#include "functions.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

const auto scale = 0.2;

const auto map_height = 261;
const auto map_width  = 301;
const auto camera_height_mm = 200;

const auto black  = Scalar(0x00, 0x00, 0x00);
const auto red    = Scalar(0x00, 0x00, 0xFF);
const auto green  = Scalar(0x00, 0xFF, 0x00);
const auto blue   = Scalar(0xFF, 0x00, 0x00);
const auto white  = Scalar(0xFF, 0xFF, 0xFF);
const auto yellow = Scalar(0x00, 0xFF, 0xFF);

const auto hue_min = 0;
const auto hue_max = 360;
const auto sat_min = 0;
const auto sat_max = 100;
const auto val_min = 0;
const auto val_max = 100;

bool detectStructuredLight(Mat &inputImage, Mat &outputImage)
{
    if (inputImage.empty() != false)
    {
        return false;
    }

    auto image = inputImage.clone();
    cvtColor(image, image, COLOR_RGB2HSV);

    auto lowerBound = vector<Scalar>();
    auto upperBound = vector<Scalar>();

    getLowerAndUpperHSVBounds(10, 350, 5, 50, 45, 100, lowerBound, upperBound);
    inRange(image, lowerBound, upperBound, image);

    auto kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(image, image, MORPH_CLOSE, kernel);
    morphologyEx(image, image, MORPH_CLOSE, kernel);

    thinnig(image, image);

    outputImage = image;

    return true;
}

bool drawMap(Mat &inputImage, Mat &outputImage)
{
    if (inputImage.empty() != false)
    {
        return false;
    }

    auto image = inputImage.clone();
    auto center = Point(image.cols / 2, image.rows / 2);

    static const auto height = static_cast<double>(image.rows);
    static const auto width  = static_cast<double>(image.cols);

    static auto angleOfViewX = 74.0 / 180.0 * CV_PI;
    static auto angleOfViewY = 2 * atan(height / width * tan(angleOfViewX / 2));

    static auto focalLengthX = width  / (2 * tan(angleOfViewX / 2));
    static auto focalLengthY = height / (2 * tan(angleOfViewY / 2));

    auto map = Mat(map_height, map_width, CV_8UC3, black);
    for (auto row = 0; row < map.rows; row++)
    {
        auto ptr = map.ptr<Vec3b>(map.rows - (row + 1));

        for (auto col = 0; col < map.cols; col++)
        {
            if (col % 20 == 0 && row % 20 == 0)
            {
                ptr[col] = Vec3b(0xFF, 0xFF, 0xFF);
            }
        }
    }

    for (auto row = 0; row < image.rows; row++)
    {
        auto ptr = image.ptr<uint8_t>(row);

        for (auto col = 0; col < image.cols; col++)
        {
            if (ptr[col] != 0xFF)
            {
                continue;
            }

            auto angleX = (col - center.x) / focalLengthX;
            auto angleY = (row - center.y) / focalLengthY;

            auto x = static_cast<uint32_t>(map.cols / 2 + camera_height_mm * tan(angleX) / tan(angleY) * scale);
            auto y = static_cast<uint32_t>(map.rows - camera_height_mm / tan(angleY) * scale);

            circle(map, Point(x, y), 1, green);
        }
    }

    outputImage = map;

    return true;
}


void waitForKeyPress(const uint32_t delay)
{
    auto key = -1;
    while (key == -1)
    {
        key = waitKey(delay);
    }
}

char waitForAnswer(const String &question)
{
    cout << endl << question;

    char key = -1;
    while (key != 'y' && key != 'n')
    {
        if (key != -1)
        {
            cout << "Please, enter your answer! ";
        }
        cin >> key;
    }

    return key;
}

void getLowerAndUpperHSVBounds(const int32_t hueLow, const int32_t hueHigh,
                               const int32_t satLow, const int32_t satHigh,
                               const int32_t valLow, const int32_t valHigh,
                               vector<Scalar> &lowerBound, vector<Scalar> &upperBound)
{
    for (auto hueCurr = hueLow, hueNext = hueHigh; hueCurr != hueNext;
         hueCurr = hueNext % (hue_max - hue_min),
         hueNext = hueHigh % (hue_max - hue_min))
    {
        if (hueCurr < hue_min)
        {
            hueCurr = hueCurr + hue_max;
            hueNext = hue_max;
        }
        if (hueNext > hue_max)
        {
            hueNext = hue_max;
        }

        for (auto satCurr = satLow, satNext = satHigh; satCurr != satNext;
             satCurr = satNext % (sat_max - sat_min),
             satNext = satHigh % (sat_max - sat_min))
        {
            if (satCurr < hue_min)
            {
                satCurr = satCurr + sat_max;
                satNext = sat_max;
            }
            if (satNext > sat_max)
            {
                satNext = sat_max;
            }

            for (auto valCurr = valLow, valNext = valHigh; valCurr != valNext;
                 valCurr = valNext % (val_max - val_min),
                 valNext = valHigh % (val_max - val_min))
            {
                if (valCurr < val_min)
                {
                    valCurr = valCurr + val_max;
                    valNext = val_max;
                }
                if (valNext > val_max)
                {
                    valNext = val_max;
                }

                lowerBound.emplace_back(Scalar(cvtHue(hueCurr), cvtSat(satCurr), cvtVal(valCurr)));
                upperBound.emplace_back(Scalar(cvtHue(hueNext), cvtSat(satNext), cvtVal(valNext)));
            }
        }
    }
}

bool thinnig(Mat &inputImage, Mat &outputImage)
{
    if (inputImage.empty() != false)
    {
        return false;
    }

    auto image = inputImage.clone();
    image /= 0xFF;

    auto isImageChanged = true;
    while (isImageChanged != false)
    {
        isImageChanged = false;
        static auto stage = Stage::STAGE_ONE;

        auto buffer = image.clone();
        auto p0 = buffer.ptr<uint8_t>(0);
        auto p1 = buffer.ptr<uint8_t>(1);

        for (auto row = 1; row < buffer.rows - 1; row++)
        {
            auto p = image.ptr<uint8_t>(row);
            auto p2 = buffer.ptr<uint8_t>(row + 1);

            for (auto col = 1; col < buffer.cols - 1; col++)
            {
                // Если пиксель не является белым,
                // переходим к следующему пикселю
                if (p1[col] != 1)
                {
                    continue;
                }

                // Если количество белых пикселей не лежит в диапазоне [2; 6],
                // переходим к следующему пикселю
                auto whitePixels = countWhitePixels(p0 + col - 1, p1 + col - 1, p2 + col - 1);
                if (whitePixels < 2 || whitePixels > 6)
                {
                    continue;
                }

                // Если количество переходов от чёрного пикселя к белому не равно 1,
                // переходим к следующему пикселю
                auto transitions = countTransitions(p0 + col - 1, p1 + col - 1, p2 + col - 1);
                if (transitions != 1)
                {
                    continue;
                }

                auto areBlack = areBorderPixelsBlack(p0 + col - 1, p1 + col - 1, p2 + col - 1, stage);
                if (areBlack != true)
                {
                    continue;
                }

                p[col] = 0;
                isImageChanged = true;
            }

            p0 = p1;
            p1 = p2;
        }

        stage == Stage::STAGE_ONE ? stage = Stage::STAGE_TWO : stage = Stage::STAGE_ONE;
    }

    outputImage = image * 0xFF;
    return true;
}

uint32_t countWhitePixels(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2)
{
    auto counter = 0;
    counter += p0[0] + p0[1] + p0[2];
    counter += p1[0] + p1[2];
    counter += p2[0] + p2[1] + p2[2];
    return counter;
}

uint32_t countTransitions(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2)
{
    auto counter = 0;
    counter += ~p0[1] & p0[2];
    counter += ~p0[2] & p1[2];
    counter += ~p1[2] & p2[2];
    counter += ~p2[2] & p2[1];
    counter += ~p2[1] & p2[0];
    counter += ~p2[0] & p1[0];
    counter += ~p1[0] & p0[0];
    counter += ~p0[0] & p0[1];
    return counter;
}

bool areBorderPixelsBlack(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2, const Stage stage)
{
    switch (stage)
    {
        case Stage::STAGE_ONE:
        {
            return !static_cast<bool>(p1[2] & p2[1] & (p0[1] | p1[0]));
        }

        case Stage::STAGE_TWO:
        {
            return !static_cast<bool>(p0[1] & p1[0] & (p1[2] | p2[1]));
        }
    }

    return false;
}
