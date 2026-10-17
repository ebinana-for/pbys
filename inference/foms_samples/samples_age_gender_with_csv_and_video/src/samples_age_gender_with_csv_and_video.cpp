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

void displayResults(cv::Mat& img, const foms::AgeGenderEstimation& estimation,
                    const foms::CropRegion& faceCrop) {

    cv::rectangle(img, cv::Point(faceCrop.xmin, faceCrop.ymin), cv::Point(faceCrop.xmax, faceCrop.ymax),
                  cv::Scalar(0, 255, 0), 2);
    std::stringstream ss1;
    std::string gender("Male");

    if (estimation.gender == 1) {
        gender = "Female";
    }

    ss1 << gender << " ; " << static_cast< int >(estimation.age);
    cv::putText(img, ss1.str(), cv::Point(faceCrop.xmin - 20, faceCrop.ymin - 20), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(0, 255, 0), 2);
}

void displayMeanResults(cv::Mat& img, const foms::AgeGenderEstimation& mean_estimation) {

    const cv::Size sz = img.size();
    cv::Mat border(50, sz.width, CV_8UC3, cv::Scalar(200, 200, 200));
    std::stringstream ss2;
    std::string gender("Male");

    if (mean_estimation.gender == 1) {
        gender = "Female";
    }

    ss2 << "Mean Gender Prediction : " << gender
        << "  Mean Age Prediction: " << static_cast< int >(mean_estimation.age)
        << "  Genre score : " << mean_estimation.score_gender
        << "  Age score : " << mean_estimation.score_age;
    cv::putText(border, ss2.str(), cv::Point(40, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    cv::vconcat(border, img, img);
}

int main(int argc, char* argv[]) {

    foms::g_logger.set_verbosity(foms::LogVerbosity::ALL);
    foms::g_logger.writeLog("Gender and Age estimation sample with video and csv", foms::errorLevel::INFO);

    ////Load the model////
    const std::string model_path = "../model-artifacts/model_age_gender";
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
    cv::namedWindow("Capture", cv::WINDOW_NORMAL);

    cv::Mat img;
    int delay = 25;
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

        if (err == foms::FomsFaceErrorType::FomsFace_NOERROR) {

            //// Estimate the age and gender /////////////////
            const auto estimation = object.estimate(img_gray, facecrop);

            ////////////////////Display//////////////////////////////////////////////////
            if (estimation.valid) {
                displayResults(img, estimation, facecrop);
            }
            delay = 5;

        } else {

            delay = 25;
        }

        //// Get consolidated estimation over frames//////////////
        const auto mean_estimation = object.get_mean_estimation();

        //////// if you want to reset the mean estimation, call  : object.reset_estimator() /////////////////

        ////////////////////Display//////////////////////////////////////////////////
        foms::drawEyesEarsShin(img, keyptLeye1, keyptReye1, keyptShin1, keyptLear1, keyptRear1,
                               cv::Scalar(0, 0, 255));
        displayMeanResults(img, mean_estimation);

        cv::resize(img, img, cv::Size(), 0.5, 0.5);

        cv::imshow("Capture", img);

        const int k = cv::waitKey(delay);
        if (k == 32) {
            cv::waitKey(0);
        } else if (k == 27) {
            break;
        }
    }
}