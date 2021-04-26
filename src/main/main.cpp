#include <iostream>
#include <cstdint>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/calib3d.hpp"
#include "functions.h"

using namespace std;
using namespace cv;

static const String videos_path = "src/videos/";

int main(int argc, char *argv[])
{
    cout << "Starting a program.";

    auto question = "Would you like to test a Structured Light [y/n]? ";
    auto key = waitForAnswer(question);

    if (key == 'y')
    {
        auto videoNames = vector<String>
        {
            "video_3.avi",
            "video_4.avi"
        };

        for (auto& videoName : videoNames)
        {
            auto video = VideoCapture(videos_path + videoName);
            auto frame = Mat();

            auto isVideoEnd = false;
            while (isVideoEnd != true)
            {
                auto isOk = video.read(frame);
                if (isOk != true)
                {
                    isVideoEnd = true;
                    continue;
                }

                auto result = Mat();
                isOk = detectStructuredLight(frame, result);
                if (isOk != true)
                {
                    continue;
                }

                auto map = Mat();
                isOk = drawMap(result, map);
                if (isOk != false)
                {
                    imshow("Original Video", frame);
                    imshow("Threshold", result);
                    imshow("Map", map);
                }

                const auto key = static_cast<char>(waitKey(1));
                if (key == 'q' || key == 'Q' || key == 27)
                {
                    isVideoEnd = true;
                }

                waitKey(30);
            }

            while (waitKey() != 27);

            destroyAllWindows();
        }
    }

    return 0;
}
