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
#include "foms_position_aggregation.hpp"
int main(int argc, char* argv[]) {

    foms::cameraParameters cameraParams;

    /// Set camera parameters
    cameraParams.distance_lateral_cameracenter_siegecenter_left = 1.0;
    cameraParams.distance_lateral_cameracenter_siegecenter_right = 1.0;
    cameraParams.distance_board_camera_long = 0.2;

    foms::g_logger.set_verbosity(foms::LogVerbosity::ALL);
    foms::g_logger.writeLog("Position estimation sample", foms::errorLevel::INFO);

    ////Load the model////
    const std::string model_path = "../model-artifacts/model_positions_device";
    foms::PositionClassification object(model_path, cameraParams);
    foms::PredictionAggregator positionAggregator;

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

    cv::Mat img;
    const int occupant_id = 0;

    cv::Mat img_gray;
    ////Iterate over the frames of the video////
    for (int index = 0;; index++) {

        video_cap >> img;
        if (img.empty()) {
            break;
        }
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

        // note if you use raw_data : pointer to uint8_t data of size rows*cols
        // uint8_t* data= raw_data;
        // cv::Mat img = cv::Mat(cv::Size(cols, rows), CV_8UC1, data);

        // set the camera roation for measures
        const double roll = 0.0 * CV_PI / 180.0;  // Example rotation angles in radians
        const double pitch = 0.0 * CV_PI / 180.0; // Example rotation angles in radians
        const double yaw = 0.0 * CV_PI / 180.0;   // Example rotation angles in radians

        const cv::Mat m_rotation = foms::getRotationMatrix(roll, pitch, yaw);

        object.set_camera_rotation(m_rotation);

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
            positionAggregator.update_observation(classifications.scores, classifications.measures);

            prediction_scores = classifications.scores;
        } else {
            const foms::positionsScores scores{};
            const foms::positionsMeasures measures{};
            positionAggregator.update_observation(scores, measures);
        }
        const auto classifications_aggregated = positionAggregator.get_aggregated_prediction();
        std::cout << index << ",";
        if (classifications_aggregated.scores.valid && classifications_aggregated.valid) {
            std::cout << classifications_aggregated.scores.closetoairbag << ","
                      << classifications_aggregated.scores.legsondashboard << ","
                      << classifications_aggregated.scores.torsodissimetry << ","
                      << classifications_aggregated.scores.reachingback << ","
                      << classifications_aggregated.scores.lateralbodycontact << ","
                      << classifications_aggregated.scores.oneshoulderincontact << ","
                      << classifications_aggregated.scores.pelvisrotation << ","
                      << classifications_aggregated.scores.pelvisdrift << "\n";

        } else {
            std::cout << "-1,-1,-1,-1,-1,-1,-1,-1"
                      << "\n";
        }
    }
}