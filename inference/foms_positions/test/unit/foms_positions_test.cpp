#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_log.hpp>
#include <boost/test/data/test_case.hpp>
#include <cstdint>
#include <vector>
#include <set>

#include "foms_positions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// // =============================================================================================
BOOST_AUTO_TEST_SUITE(_main_positions_class);
// // =============================================================================================

BOOST_AUTO_TEST_CASE(_empty_positions_object) {
    const foms::PositionClassification object{};
    BOOST_CHECK(object.is_loaded() == false);
}
BOOST_AUTO_TEST_CASE(_falsepath_ti_dlr_object) {
    foms::PositionClassification object{"test.xml"};
    BOOST_CHECK(object.is_loaded() == false);

    const std::string image_path = "../../../data/images/1.png";
    cv::Mat image{cv::imread(image_path)};
    BOOST_CHECK(image.rows != 0);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    const foms::CropRegion valid_crop(0, 224, 0, 224); // Invalid crop region
    const std::vector< foms::BodyKeyPoint > bodyKeyPts(28, foms::BodyKeyPoint(0, 0, 0.0, 0));
    const auto ret1 = object.classify(image, valid_crop, bodyKeyPts);
    BOOST_CHECK(ret1.valid == false);

    const auto ret2 = object.classify(image.data);
    BOOST_CHECK(ret2.valid == false);
}

BOOST_AUTO_TEST_CASE(_positions_object_model_loaded) {
    const foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
}

BOOST_AUTO_TEST_CASE(_positions_object_null_ptr) {
    const foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    uint8_t* data = nullptr;
    const auto ret = object.classify(data);
    BOOST_CHECK(ret.valid == false);
}

BOOST_AUTO_TEST_CASE(_positions_object_inference) {
    const foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    const std::string image_path = "../../../data/images/1.png";
    cv::Mat image{cv::imread(image_path)};
    BOOST_CHECK(image.rows != 0);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(224, 224));

    const auto ret = object.classify(image.data);

    BOOST_CHECK(ret.valid == true);
}

BOOST_AUTO_TEST_CASE(_positions_object_invalid_image_type) {
    foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    const std::string image_path = "../../../data/images/1.png";
    const cv::Mat image{cv::imread(image_path)};
    BOOST_CHECK(image.rows != 0);

    cv::Mat invalid_image;
    image.convertTo(invalid_image, CV_32FC3); // Invalid image type

    const foms::CropRegion crop(0, 192, 0, 256);
    const std::vector< foms::BodyKeyPoint > bodyKeyPts(28, foms::BodyKeyPoint(0, 0, 0.0, 0));
    const auto ret = object.classify(invalid_image, crop, bodyKeyPts);

    BOOST_CHECK(ret.valid == false);
}

BOOST_AUTO_TEST_CASE(_positions_object_invalid_crop) {
    foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    const std::string image_path = "../../../data/images/1.png";
    cv::Mat image{cv::imread(image_path)};
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    const foms::CropRegion invalid_crop(0, 500, 0, 500); // Invalid crop region
    const std::vector< foms::BodyKeyPoint > bodyKeyPts(28, foms::BodyKeyPoint(0, 0, 0.0, 0));
    const auto ret = object.classify(image, invalid_crop, bodyKeyPts);

    BOOST_CHECK(ret.valid == false);
}

BOOST_AUTO_TEST_CASE(_positions_object_valid_crop) {
    foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    const std::string image_path = "../../../data/images/1.png";
    cv::Mat image{cv::imread(image_path)};

    const foms::CropRegion valid_crop(0, 192, 0, 256); // valid crop region
    const std::vector< foms::BodyKeyPoint > bodyKeyPts(28, foms::BodyKeyPoint(0, 0, 0.0, 0));

    const auto ret0 = object.classify(image, valid_crop, bodyKeyPts);

    BOOST_CHECK(ret0.valid == false);

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    const auto ret1 = object.classify(image, valid_crop, bodyKeyPts);

    BOOST_CHECK(ret1.valid == true);

    const auto ret2 =
        object.classify(image.data, image.cols, image.rows, image.channels(), valid_crop, bodyKeyPts);

    BOOST_CHECK(ret2.valid == true);

    const auto ret3 = object.classify(image.data, image.cols, image.rows, 2, valid_crop, bodyKeyPts);

    BOOST_CHECK(ret3.valid == false);
}

BOOST_AUTO_TEST_CASE(_positions_object_invalid_body_keypoints) {
    foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    const std::string image_path = "../../../data/images/0.png";
    cv::Mat image{cv::imread(image_path)};
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    const foms::CropRegion crop(0, 192, 0, 256);
    const std::vector< foms::BodyKeyPoint > invalid_bodyKeyPts(
        10, foms::BodyKeyPoint(0, 0, 0.0, 0)); // Invalid number of keypoints
    const auto ret = object.classify(image, crop, invalid_bodyKeyPts);

    BOOST_CHECK(ret.valid == false);
}

BOOST_AUTO_TEST_CASE(_positions_measures) {
    foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    const foms::CsvReader reader("../../../data/example.csv", ',');
    BOOST_CHECK(reader.is_loaded());

    std::vector< foms::BodyKeyPoint > bodyKeyPts;
    bodyKeyPts.reserve(foms::NumberOfBodyKeypoints);

    for (int keyptsIndex = 0; keyptsIndex < foms::NumberOfBodyKeypoints; keyptsIndex++) {
        bodyKeyPts.emplace_back(reader, 0, 0, keyptsIndex);
    }
    const auto distance = object.compute_distance_from_board(bodyKeyPts);
    BOOST_CHECK(distance > 0.0);

    const auto rotation_measure = object.compute_measures_from_keypoints(bodyKeyPts);
    BOOST_CHECK(rotation_measure.shoulderOrientation > 0.0);

    const auto rotation = cv::Mat::eye(3, 3, CV_64F); // Identity rotation matrix
    object.set_camera_rotation(rotation);

    bodyKeyPts[foms::NoseKeypointIndex].valid = false; // Invalidating the nose keypoint
    const auto invalid_distance = object.compute_distance_from_board(bodyKeyPts);
    BOOST_CHECK(invalid_distance == 0.0); // Expecting a negative distance when nose keypoint is invalid

    bodyKeyPts[foms::LeftShoulderKeypointIndex].valid = false;
    const auto rotation_measure_invalid = object.compute_measures_from_keypoints(bodyKeyPts);
    BOOST_CHECK(rotation_measure_invalid.shoulderOrientation < 0.0);
}

BOOST_AUTO_TEST_CASE(_positions_inference) {
    foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    const foms::CsvReader reader("../../../data/example.csv", ',');
    BOOST_CHECK(reader.is_loaded());

    const std::string image_path = "../../../data/images/2.png";
    cv::Mat image{cv::imread(image_path)};
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    const foms::BodyBBox bbox(reader, 0, 0);
    BOOST_CHECK(bbox.valid);
    std::vector< foms::BodyKeyPoint > bodyKeyPts;
    bodyKeyPts.reserve(foms::NumberOfBodyKeypoints);

    for (int keyptsIndex = 0; keyptsIndex < foms::NumberOfBodyKeypoints; keyptsIndex++) {
        bodyKeyPts.emplace_back(reader, 0, 0, keyptsIndex);
    }

    const foms::CropRegion crop = bbox.getCrop();
    BOOST_CHECK(crop.isvalid(image.cols, image.rows));
    const auto ret = object.classify(image, crop, bodyKeyPts);
    BOOST_CHECK(ret.valid == true);

    const auto ret2 = object.estimate(image, crop, bodyKeyPts);
    BOOST_CHECK(ret2.valid == true);

    foms::drawLineKeyPts(image, bodyKeyPts);
    BOOST_CHECK(image.rows > 0 && image.cols > 0);
}

BOOST_AUTO_TEST_CASE(_positions_object_rotation) {
    foms::PositionClassification object{"../../../model-artifacts/model_positions"};
    BOOST_CHECK(object.is_loaded());
    object.set_camera_rotation(cv::Mat::eye(2, 2, CV_64F));
    cv::Mat rotation = object.get_camera_rotation();
    BOOST_CHECK(rotation.rows == 3 && rotation.cols == 3);

    object.set_camera_rotation(cv::Mat::zeros(3, 3, CV_64F));
    rotation = object.get_camera_rotation();
    BOOST_CHECK(rotation.rows == 3 && rotation.cols == 3);
    BOOST_CHECK(rotation.at< double >(0, 0) == 0.0);
    BOOST_CHECK(object.is_loaded());
}
BOOST_AUTO_TEST_SUITE_END();
