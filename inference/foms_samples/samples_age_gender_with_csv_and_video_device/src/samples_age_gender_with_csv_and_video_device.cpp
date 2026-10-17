#include "foms_age_gender_estimator.hpp"
#include "foms_logStream.hpp"
#include "foms_csv_reader.hpp"
#include <sstream>
// clang-format off

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// clang-format on
#include "foms_ti_dlr_inference.hpp"
#include "foms_utils.hpp"

int main(int argc, char* argv[]) {

    foms::g_logger.set_verbosity(foms::LogVerbosity::ALL);
    foms::g_logger.writeLog("Gender and Age estimation sample with video and csv", foms::errorLevel::INFO);

    ////Load the model////
    const std::string model_path = "../model-artifacts/model_age_gender_device";
    foms::AgeGenderEstimator object(model_path);

    std::string csv_filename = "../data/videos/thibault_vincent_28_M_178_68.csv";
    std::string video_filename = "../data/videos/thibault_vincent_28_M_178_68.mp4";

    if (argc == 3) {
        video_filename = argv[1];
        csv_filename = argv[2];
    } else if (argc > 1) {
        std::stringstream ss;
        ss << "Usage: " << argv[0] << " videofile csvfile ";
        foms::g_logger.writeLog(ss.str(), foms::errorLevel::ERROR);
        return 0;
    }

    ///////Load the csv file containing the smart eye keypoints data///
    const foms::CsvReader reader(csv_filename);

    if (!reader.is_loaded()) {
        foms::g_logger.writeLog("Unable to load csv file , exit", foms::errorLevel::ERROR);
        return 0;
    }

    if (!object.is_loaded()) {
        foms::g_logger.writeLog("Unable to load model , exit", foms::errorLevel::ERROR);
        return 0;
    }

    ///////Load the video////////
    cv::VideoCapture video_cap(video_filename);

    if (!video_cap.isOpened()) {
        foms::g_logger.writeLog("Unable to load video , exit", foms::errorLevel::ERROR);
        return 0;
    }

    cv::Mat img;

    const int occupant_id = 0;
    cv::Mat img_gray;
    ///////Iterate over frames////////
    for (int index = 0;; index++) {

        video_cap >> img;

        if (img.empty()) {
            break;
        }
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

        // note if you use raw_data : pointer to uint8_t data of size rows*cols
        // uint8_t* data= raw_data;
        // cv::Mat img = cv::Mat(cv::Size(cols, rows), CV_8UC1, data);

        ////Load the keypoints from the csv reader for the current frame and current id /////////////////
        ////!!!!!!!!!!!!!!!!!!!!!!!WARNING Left and Right corresponds to the Left and Right in image, not
        /// the left and right from the person point of view

        const foms::BodyKeyPoint keyptShin1(reader, index, occupant_id, foms::ShinKeypointsIndex);
        const foms::BodyKeyPoint keyptLear1(reader, index, occupant_id, foms::LeftEarKeypointIndex);
        const foms::BodyKeyPoint keyptRear1(reader, index, occupant_id, foms::RightEarKeypointIndex);
        const foms::BodyKeyPoint keyptLeye1(reader, index, occupant_id, foms::LeftEyeKeypointIndex);
        const foms::BodyKeyPoint keyptReye1(reader, index, occupant_id, foms::RightEyeKeypointIndex);

        //// Compute the face crop from the keypoints /////////////////
        foms::CropRegion facecrop{0, 224, 0, 224};

        const auto err = foms::computeFaceCropFromkeyPoints(keyptLeye1, keyptReye1, keyptShin1, keyptLear1,
                                                            keyptRear1, facecrop);

        //////// if you want to reset the mean estimation, call  : object.reset_estimator() /////////////////

        foms::AgeGenderEstimation estimation{};
        foms::AgeGenderEstimation mean_estimation{};

        if (err == foms::FomsFaceErrorType::FomsFace_NOERROR) {
            //// Estimate the age and gender /////////////////
            estimation = object.estimate(img_gray, facecrop);
        }
        //// Get consolidated estimation over frames//////////////
        mean_estimation = object.get_mean_estimation();
        std::cout << index << ",";
        if (estimation.valid && mean_estimation.valid) {
            std::cout << static_cast< int >(estimation.age) << ",";
            std::cout << static_cast< int >(estimation.gender) << ",";
            std::cout << estimation.score_gender << ",";
            std::cout << static_cast< int >(mean_estimation.age) << ","
                      << static_cast< int >(mean_estimation.gender) << "," << mean_estimation.score_age << ","
                      << mean_estimation.score_gender << "\n";
        } else {
            std::cout << "-1,-1,-1,-1,-1,-1,-1"
                      << "\n";
        }
    }
}