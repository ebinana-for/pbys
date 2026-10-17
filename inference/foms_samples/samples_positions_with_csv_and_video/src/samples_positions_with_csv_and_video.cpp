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

void displayResults(cv::Mat& img, const foms::positionsResult& classifications,
                    const foms::positionsResult& classifications_aggregated) {
    const cv::Size sz = img.size();
    cv::Mat border(sz.height, 800, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto font_color = cv::Scalar(255, 0, 0);
    const auto border_color = cv::Scalar(255, 0, 0);

    const auto color_red = cv::Scalar(0, 0, 255);
    const auto color_green = cv::Scalar(0, 255, 0);
    cv::Scalar color{};

    {
        if (classifications_aggregated.closetoairbag) {
            color = color_red;
        } else {
            color = color_green;
        }
        cv::rectangle(border, cv::Point(0, 0), cv::Point(800, 80), color, -1);
        cv::rectangle(border, cv::Point(0, 0), cv::Point(800, 80), border_color, 1);
        std::stringstream ss;
        ss << "CLOSE TO AIRBAG : " << (classifications_aggregated.closetoairbag ? "True" : "False")
           << "  Confidence: " << classifications_aggregated.scores.closetoairbag;
        cv::putText(border, ss.str(), cv::Point(40, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
        ss.str("");
        ss.clear();
        ss << "Distance: " << classifications.measures.dashboardDistance
           << " Framewise Confidence: " << classifications.scores.closetoairbag;
        cv::putText(border, ss.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
    }

    {
        if (classifications_aggregated.pelvisrotation) {
            color = color_red;
        } else {
            color = color_green;
        }
        cv::rectangle(border, cv::Point(0, 80), cv::Point(800, 160), color, -1);
        cv::rectangle(border, cv::Point(0, 80), cv::Point(800, 160), border_color, 1);
        std::stringstream ss;
        ss << "PELVIS ROTATION : " << (classifications_aggregated.pelvisrotation ? "True" : "False")
           << "  Confidence: " << classifications_aggregated.scores.pelvisrotation;
        cv::putText(border, ss.str(), cv::Point(40, 110), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
        ss.str("");
        ss.clear();
        ss << "Distance: "
           << " 0 "
           << " Framewise Confidence: " << classifications.scores.pelvisrotation;
        cv::putText(border, ss.str(), cv::Point(50, 140), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
    }

    {
        if (classifications_aggregated.torsodissimetry) {
            color = color_red;
        } else {
            color = color_green;
        }
        cv::rectangle(border, cv::Point(0, 160), cv::Point(800, 240), color, -1);
        cv::rectangle(border, cv::Point(0, 160), cv::Point(800, 240), border_color, 1);
        std::stringstream ss;
        ss << "TORSO DISSIMETRY : " << (classifications_aggregated.torsodissimetry ? "True" : "False")
           << "  Confidence: " << classifications_aggregated.scores.torsodissimetry;
        cv::putText(border, ss.str(), cv::Point(40, 190), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
        ss.str("");
        ss.clear();
        ss << "Distance: " << classifications.measures.shoulderOrientation
           << " Framewise Confidence: " << classifications.scores.torsodissimetry;
        cv::putText(border, ss.str(), cv::Point(50, 220), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
    }

    {
        if (classifications_aggregated.oneshoulderincontact) {
            color = color_red;
        } else {
            color = color_green;
        }
        cv::rectangle(border, cv::Point(0, 240), cv::Point(800, 320), color, -1);
        cv::rectangle(border, cv::Point(0, 240), cv::Point(800, 320), border_color, 1);
        std::stringstream ss;
        ss << "ONE SHOULDER IN CONTACT : "
           << (classifications_aggregated.oneshoulderincontact ? "True" : "False")
           << "  Confidence: " << classifications_aggregated.scores.oneshoulderincontact;
        cv::putText(border, ss.str(), cv::Point(40, 270), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
        ss.str("");
        ss.clear();
        ss << "Distance: "
           << " 0 "
           << " Framewise Confidence: " << classifications.scores.oneshoulderincontact;
        cv::putText(border, ss.str(), cv::Point(50, 300), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
    }

    {
        if (classifications_aggregated.legsondashboard) {
            color = color_red;
        } else {
            color = color_green;
        }
        cv::rectangle(border, cv::Point(0, 320), cv::Point(800, 400), color, -1);
        cv::rectangle(border, cv::Point(0, 320), cv::Point(800, 400), border_color, 1);
        std::stringstream ss;
        ss << "LEGS ON DASHBOARD : " << (classifications_aggregated.legsondashboard ? "True" : "False")
           << "  Confidence: " << classifications_aggregated.scores.legsondashboard;
        cv::putText(border, ss.str(), cv::Point(40, 350), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
        ss.str("");
        ss.clear();
        ss << "Distance: "
           << " 0 "
           << " Framewise Confidence: " << classifications.scores.legsondashboard;
        cv::putText(border, ss.str(), cv::Point(50, 380), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
    }

    {
        if (classifications_aggregated.reachingback) {
            color = color_red;
        } else {
            color = color_green;
        }
        cv::rectangle(border, cv::Point(0, 400), cv::Point(800, 480), color, -1);
        cv::rectangle(border, cv::Point(0, 400), cv::Point(800, 480), border_color, 1);
        std::stringstream ss;
        ss << "REACHING BACK : " << (classifications_aggregated.reachingback ? "True" : "False")
           << "  Confidence: " << classifications_aggregated.scores.reachingback;
        cv::putText(border, ss.str(), cv::Point(40, 430), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
        ss.str("");
        ss.clear();
        ss << "Distance: "
           << " 0 "
           << " Framewise Confidence: " << classifications.scores.reachingback;
        cv::putText(border, ss.str(), cv::Point(50, 460), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
    }

    {
        if (classifications_aggregated.lateralbodycontact) {
            color = color_red;
        } else {
            color = color_green;
        }
        cv::rectangle(border, cv::Point(0, 480), cv::Point(800, 560), color, -1);
        cv::rectangle(border, cv::Point(0, 480), cv::Point(800, 560), border_color, 1);
        std::stringstream ss;
        ss << "LATERAL BODY CONTACT : " << (classifications_aggregated.lateralbodycontact ? "True" : "False")
           << "  Confidence: " << classifications_aggregated.scores.lateralbodycontact;
        cv::putText(border, ss.str(), cv::Point(40, 510), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
        ss.str("");
        ss.clear();
        ss << "Distance: "
           << " 0 "
           << " Framewise Confidence: " << classifications.scores.lateralbodycontact;
        cv::putText(border, ss.str(), cv::Point(50, 540), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
    }

    {
        if (classifications_aggregated.pelvisdrift) {
            color = color_red;
        } else {
            color = color_green;
        }
        cv::rectangle(border, cv::Point(0, 560), cv::Point(800, 640), color, -1);
        cv::rectangle(border, cv::Point(0, 560), cv::Point(800, 640), border_color, 1);
        std::stringstream ss;
        ss << "PELVIS DRIFT: " << (classifications_aggregated.pelvisdrift ? "True" : "False")
           << "  Confidence: " << classifications_aggregated.scores.pelvisdrift;
        cv::putText(border, ss.str(), cv::Point(40, 590), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
        ss.str("");
        ss.clear();
        ss << "Distance: "
           << " 0 "
           << " Framewise Confidence: " << classifications.scores.pelvisdrift;
        cv::putText(border, ss.str(), cv::Point(50, 620), cv::FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2);
    }
    cv::hconcat(img, border, img);
}

int main(int argc, char* argv[]) {

    foms::cameraParameters cameraParams;

    /// Set camera parameters
    cameraParams.distance_lateral_cameracenter_siegecenter_left = 1.0;
    cameraParams.distance_lateral_cameracenter_siegecenter_right = 1.0;
    cameraParams.distance_board_camera_long = 0.2;

    foms::g_logger.set_verbosity(foms::LogVerbosity::ALL);
    foms::g_logger.writeLog("Position estimation sample", foms::errorLevel::INFO);

    ////Load the model////
    const std::string model_path = "../model-artifacts/model_positions_without_keypts";
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

    cv::namedWindow("Capture", cv::WINDOW_NORMAL);

    cv::Mat img;
    const int occupant_id = 0;
    int delay = 25;
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
            const auto classifications = object.estimate(img_gray, crop, bodyKeyPts);
            positionAggregator.update_observation(classifications.scores, classifications.measures);

            const auto classifications_aggregated = positionAggregator.get_aggregated_prediction();

            ////////////////////Display//////////////////////////////////////////////////

            displayResults(img, classifications, classifications_aggregated);

            cv::rectangle(img, cv::Point(bbox.x, bbox.y),
                          cv::Point(bbox.x + bbox.width, bbox.y + bbox.height), cv::Scalar(255, 0, 0));

            foms::drawLineKeyPts(img, bodyKeyPts);
            delay = 5;

        } else {
            const foms::positionsScores scores{};
            const foms::positionsMeasures measures{};
            positionAggregator.update_observation(scores, measures);
            delay = 25;
        }

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