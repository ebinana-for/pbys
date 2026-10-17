#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_log.hpp>
#include <boost/test/data/test_case.hpp>
#include <cstdint>
#include <vector>
#include <set>

#include "foms_age_gender_estimator.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// // =============================================================================================
BOOST_AUTO_TEST_SUITE(_main_age_gender_class);
// // =============================================================================================

BOOST_AUTO_TEST_CASE(_empty_age_gender_object) {
    foms::AgeGenderEstimator object{};
    BOOST_CHECK(object.is_loaded() == false);
    uint8_t* data = nullptr;
    const auto ret = object.estimate(data);
    BOOST_CHECK(ret.valid == false);
}
BOOST_AUTO_TEST_CASE(_falsepath_ti_dlr_object) {
    foms::AgeGenderEstimator object{"test.xml"};
    BOOST_CHECK(object.is_loaded() == false);
    const std::string image_path = "../../../data/images/0.png";
    cv::Mat image{cv::imread(image_path)};
    BOOST_CHECK(image.rows != 0);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    const foms::CropRegion crop{0, 192, 0, 224};
    const auto ret1 = object.estimate(image, crop);
    BOOST_CHECK(ret1.valid == false);

    const auto ret2 = object.estimate(image.data);
    BOOST_CHECK(ret2.valid == false);
}
BOOST_AUTO_TEST_CASE(_age_gender_object_model_loaded) {
    const foms::AgeGenderEstimator object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded());
}

BOOST_AUTO_TEST_CASE(_age_gender_object_null_ptr) {
    foms::AgeGenderEstimator object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded());
    uint8_t* data = nullptr;
    const auto ret = object.estimate(data);
    BOOST_CHECK(ret.valid == false);
}

BOOST_AUTO_TEST_CASE(_age_gender_object_inference) {
    foms::AgeGenderEstimator object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded());
    const std::string image_path = "../../../data/images/0.png";
    cv::Mat image{cv::imread(image_path)};
    BOOST_CHECK(image.rows != 0);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    image = image(cv::Rect(0, 0, 192, 224)).clone();
    const auto ret = object.estimate(image.data);
    BOOST_CHECK(ret.valid == true);

    BOOST_CHECK(ret.gender == 0 && ret.age > 20 && ret.age < 40);
}

BOOST_AUTO_TEST_CASE(_age_gender_object_inference_image) {
    foms::AgeGenderEstimator object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded());
    const std::string image_path = "../../../data/images/0.png";
    cv::Mat image{cv::imread(image_path)};
    BOOST_CHECK(image.rows != 0);

    foms::CropRegion crop{0, 192, 0, 224};
    const auto ret0 = object.estimate(image, crop);
    BOOST_CHECK(ret0.valid == false);

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    const auto mean = object.get_mean_estimation();
    BOOST_CHECK(mean.valid == false);

    const auto ret1 = object.estimate(image, crop);
    BOOST_CHECK(ret1.valid == true);

    BOOST_CHECK(ret1.gender == 0 && ret1.age > 20 && ret1.age < 40);

    const auto ret1_mean = object.get_mean_estimation();
    BOOST_CHECK(ret1_mean.valid == true);

    crop.xmax = 178;

    const auto ret2 = object.estimate(image, crop);
    BOOST_CHECK(ret2.valid == true);

    crop.xmax = 224;

    const auto ret3 = object.estimate(image, crop);
    BOOST_CHECK(ret3.valid == false);
}

BOOST_AUTO_TEST_CASE(_age_gender_object_rotation) {
    foms::AgeGenderEstimator object{"../../../model-artifacts/model_age_gender"};
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
