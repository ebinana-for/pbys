#include "foms_positions.hpp"
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

    const foms::cameraParameters cameraParams;
    foms::g_logger.set_verbosity(foms::LogVerbosity::ALL);
    foms::g_logger.writeLog("Positions evaluation", foms::errorLevel::INFO);

    ////Load the model////

    const std::string model_path = "../model-artifacts/model_positions_data_aug/";
    foms::PositionClassification object(model_path, cameraParams);

    if (!object.is_loaded()) {
        foms::g_logger.writeLog("Unable to load model , exit", foms::errorLevel::ERROR);
        return 0;
    }
    std::string directory = "../data/videos/";
    if (argc == 2) {
        directory = argv[1];
    } else if (argc > 2) {
        std::stringstream ss;
        ss << "Usage: " << argv[0] << " directory";
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
            video_filename.substr(0, video_filename.find_last_of('.')) + "_out_positions.csv";
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
        out_csv_file << "index,closetoairbag,legsondashboard,torsodissimetry,reachingback,lateralbodycontact,"
                        "oneshoulderincontact,pelvisrotation,pelvisdrift"
                     << "\n";

        ///////Load the video////////
        cv::VideoCapture video_cap(video_filename);

        cv::Mat img;
        const int occupant_id = 0;

        cv::Mat img_gray;
        ////Iterate over the frames of the video////
        for (int index = 0;; index++) {

            video_cap >> img;
            if (img.empty()) {
                break;
            }
            out_csv_file << index << ",";

            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
            ////check if the bbox is valid and get the crop region and the body keypoints////
            //// To be replace when not using csv by const foms::BodyBBox bbox(x,y, w, h,valid_,id);
            const foms::BodyBBox bbox(reader, index, occupant_id);
            foms::positionsResult classifications{};
            foms::positionsScores prediction_scores{};

            if (bbox.valid) {
                // crop correspondant à la bbox de l'occupant
                const foms::CropRegion crop = bbox.getCrop();

                // fill body keypoints;
                std::vector< foms::BodyKeyPoint > bodyKeyPts;
                bodyKeyPts.reserve(foms::NumberOfBodyKeypoints);

                for (int keyptsIndex = 0; keyptsIndex < foms::NumberOfBodyKeypoints; keyptsIndex++) {
                    bodyKeyPts.emplace_back(reader, index, occupant_id, keyptsIndex);
                    /* To be replaced by :
                        bodyKeyPts.emplace_back(x,y,occupant_id,quality);
                        with x,y,quality corresponding to the key point with index keyptsIndex
                    */
                }

                //// Classify the positions /////////////////
                classifications = object.estimate(img_gray, crop, bodyKeyPts);

                prediction_scores = classifications.scores;
            }

            if (prediction_scores.valid && classifications.valid) {
                out_csv_file << prediction_scores.closetoairbag << "," << prediction_scores.legsondashboard
                             << "," << prediction_scores.torsodissimetry << ","
                             << prediction_scores.reachingback << "," << prediction_scores.lateralbodycontact
                             << "," << prediction_scores.oneshoulderincontact << ","
                             << prediction_scores.pelvisrotation << "," << prediction_scores.pelvisdrift
                             << "\n";

            } else {
                out_csv_file << "-1,-1,-1,-1,-1,-1,-1,-1"
                             << "\n";
            }
        }

        video_cap.release();
        out_csv_file.close();
        foms::g_logger.writeLog("End of video and csv processing", foms::errorLevel::INFO);
    }
}
