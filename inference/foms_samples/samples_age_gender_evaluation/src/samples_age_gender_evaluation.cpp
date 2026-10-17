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

#include <filesystem>
namespace fs = std::filesystem;

void find_files(const std::string& directory, std::vector< std::string >& csv_filename_list,
                std::vector< std::string >& mp4_filename_list) {
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {

        if (entry.path().extension() == ".mp4") {
            const std::string mp4_filename = entry.path();
            std::string csv_filename = mp4_filename;
            csv_filename.replace(mp4_filename.find(".mp4"), 4, ".csv");

            if (fs::exists(csv_filename)) {
                std::cout << "Found mp4 file: " << mp4_filename << "\n";
                std::cout << "Found csv file: " << csv_filename << "\n";
                mp4_filename_list.emplace_back(mp4_filename);
                csv_filename_list.emplace_back(csv_filename);
            }
        }
    }
}

int main(int argc, char* argv[]) {

    foms::g_logger.set_verbosity(foms::LogVerbosity::ALL);
    foms::g_logger.writeLog("Gender and Age estimation sample with video and csv", foms::errorLevel::INFO);

    ////Load the model////
    const std::string model_path = "../model-artifacts/model_age_gender";
    foms::AgeGenderEstimator object(model_path);

    if (!object.is_loaded()) {
        foms::g_logger.writeLog("Unable to load model , exit", foms::errorLevel::ERROR);
        return 0;
    }

    std::string directory = "../data/videos/";
    if (argc == 2) {
        directory = argv[1];
    } else if (argc > 2) {
        std::stringstream ss;
        ss << "Usage: " << argv[0] << " directory ";
        foms::g_logger.writeLog(ss.str(), foms::errorLevel::ERROR);
        return 0;
    }

    std::vector< std::string > csv_filename_list{};
    std::vector< std::string > mp4_filename_list{};

    find_files(directory, csv_filename_list, mp4_filename_list);

    for (size_t i = 0; i < mp4_filename_list.size(); i++) {

        std::cout << "Processing video: " << mp4_filename_list[i] << "\n";
        std::cout << "Processing csv: " << csv_filename_list[i] << "\n";
        const std::string csv_filename = csv_filename_list[i];
        const std::string video_filename = mp4_filename_list[i];
        const std::string out_csv_filename =
            video_filename.substr(0, video_filename.find_last_of('.')) + "_out_age.csv";
        std::cout << "Output csv: " << out_csv_filename << "\n";

        ///////Load the csv file containing the smart eye keypoints data///
        const foms::CsvReader reader(csv_filename);

        if (!reader.is_loaded()) {
            foms::g_logger.writeLog("Unable to load csv file , exit", foms::errorLevel::ERROR);
            return 0;
        }

        std::ofstream out_csv_file;
        out_csv_file.open(out_csv_filename);

        if (!out_csv_file.is_open()) {
            foms::g_logger.writeLog("Unable to open output csv file , exit", foms::errorLevel::ERROR);
            return 0;
        }
        out_csv_file << "index"
                     << ","
                     << "age"
                     << ","
                     << "gender"
                     << ","
                     << "score_gender"
                     << ","
                     << "age_mean"
                     << ","
                     << "gender_mean"
                     << ","
                     << "score_age_mean"
                     << ","
                     << "score_gender_mean"
                     << "\n";

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

            const auto err = foms::computeFaceCropFromkeyPoints(keyptLeye1, keyptReye1, keyptShin1,
                                                                keyptLear1, keyptRear1, facecrop);
            out_csv_file << index << ",";

            foms::AgeGenderEstimation estimation{};
            foms::AgeGenderEstimation mean_estimation{};

            if (err == foms::FomsFaceErrorType::FomsFace_NOERROR) {
                //// Estimate the age and gender /////////////////
                estimation = object.estimate(img_gray, facecrop);
            }
            //// Get consolidated estimation over frames//////////////
            mean_estimation = object.get_mean_estimation();

            if (estimation.valid && mean_estimation.valid) {
                out_csv_file << static_cast< int >(estimation.age) << ",";
                out_csv_file << static_cast< int >(estimation.gender) << ",";
                out_csv_file << estimation.score_gender << ",";
                out_csv_file << static_cast< int >(mean_estimation.age) << ","
                             << static_cast< int >(mean_estimation.gender) << "," << mean_estimation.score_age
                             << "," << mean_estimation.score_gender << "\n";

            } else {
                out_csv_file << "-1,-1,-1,-1,-1,-1,-1"
                             << "\n";
            }
        }
        out_csv_file.close();

        foms::g_logger.writeLog("End of video and csv processing", foms::errorLevel::INFO);
        object.reset_estimator();
    }
}