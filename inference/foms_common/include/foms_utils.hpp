#ifndef FOMS_UTILS_HPP
#define FOMS_UTILS_HPP

#include <vector>
#include <array>
#include <map>
#include <string>
#include <iostream>
#include <sys/time.h>
#include <stdint.h>
#include <sstream>
#include "foms_csv_reader.hpp"
#include "foms_logStream.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
 * @file foms_utils.hpp
 * @brief Utility functions and structures for FOMS .
 *
 * This file contains various utility functions, constants, and structures used in the FOMS library.
 *
 */

namespace foms {

/**
 * @brief Constants for occupant keypoints and bounding boxes field names.
 */
inline constexpr char occupantskeypointsName[] = "/trackingState/occupants/";
inline constexpr char occupantsbboxName[] = "/trackingState/boundingBoxes/";

inline constexpr char occupantsLeftEyeValidName[] = "/leftEye/valid";
inline constexpr char occupantsLeftEyeXName[] = "/leftEye/upperLeft/0";
inline constexpr char occupantsLeftEyeYName[] = "/leftEye/upperLeft/1";
inline constexpr char occupantsLeftEyeWName[] = "/leftEye/width";
inline constexpr char occupantsLeftEyeHName[] = "/leftEye/height";

inline constexpr char occupantsRightEyeValidName[] = "/rightEye/valid";
inline constexpr char occupantsRightEyeXName[] = "/rightEye/upperLeft/0";
inline constexpr char occupantsRightEyeYName[] = "/rightEye/upperLeft/1";
inline constexpr char occupantsRightEyeWName[] = "/rightEye/width";
inline constexpr char occupantsRightEyeHName[] = "/rightEye/height";

inline constexpr char bodykeypointName[] = "/bodyKeyPoints/";
inline constexpr char keypointXName[] = "/value/0";
inline constexpr char keypointYName[] = "/value/1";
inline constexpr char keypointQualityName[] = "/quality";

inline constexpr char occupantsBBoxValidName[] = "/boundingBox/valid";
inline constexpr char occupantsBBoxXName[] = "/boundingBox/upperLeft/0";
inline constexpr char occupantsBBoxYName[] = "/boundingBox/upperLeft/1";
inline constexpr char occupantsBBoxWName[] = "/boundingBox/width";
inline constexpr char occupantsBBoxHName[] = "/boundingBox/height";

inline std::string getBodykeypointXFieldName(int occupant_id, int keypointNumber) {
    std::stringstream ss;
    ss << occupantskeypointsName << occupant_id << bodykeypointName << keypointNumber << keypointXName;
    return ss.str();
};

inline std::string getBodykeypointYFieldName(int occupant_id, int keypointNumber) {
    std::stringstream ss;
    ss << occupantskeypointsName << occupant_id << bodykeypointName << keypointNumber << keypointYName;
    return ss.str();
};

inline std::string getBodykeypointQualityFieldName(int occupant_id, int keypointNumber) {
    std::stringstream ss;
    ss << occupantskeypointsName << occupant_id << bodykeypointName << keypointNumber << keypointQualityName;
    return ss.str();
};

inline std::string getLeftkeypointQualityFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsLeftEyeValidName;
    return ss.str();
};

inline std::string getLeftkeypointXFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsLeftEyeXName;
    return ss.str();
};

inline std::string getLeftkeypointYFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsLeftEyeYName;
    return ss.str();
};

inline std::string getLeftkeypointWFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsLeftEyeWName;
    return ss.str();
};

inline std::string getLeftkeypointHFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsLeftEyeHName;
    return ss.str();
};

inline std::string getRightkeypointQualityFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsRightEyeValidName;
    return ss.str();
};

inline std::string getRightkeypointXFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsRightEyeXName;
    return ss.str();
};

inline std::string getRightkeypointYFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsRightEyeYName;
    return ss.str();
};

inline std::string getRightkeypointWFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsRightEyeWName;
    return ss.str();
};

inline std::string getRightkeypointHFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantsbboxName << occupant_id << occupantsRightEyeHName;
    return ss.str();
};

inline std::string getBBoxValidFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantskeypointsName << occupant_id << occupantsBBoxValidName;
    return ss.str();
};

inline std::string getBBoxXFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantskeypointsName << occupant_id << occupantsBBoxXName;
    return ss.str();
};

inline std::string getBBoxYFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantskeypointsName << occupant_id << occupantsBBoxYName;
    return ss.str();
};

inline std::string getBBoxWFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantskeypointsName << occupant_id << occupantsBBoxWName;
    return ss.str();
};

inline std::string getBBoxHFieldName(int occupant_id) {
    std::stringstream ss;
    ss << occupantskeypointsName << occupant_id << occupantsBBoxHName;
    return ss.str();
};

inline constexpr int NumberOfBodyKeypoints = 28;
inline constexpr int NumberOfNeededBodyKeypoints = 16;
inline constexpr int NumberOfLineKeypoints = 10;
inline constexpr int NoseKeypointIndex = 0;
inline constexpr int LeftEyeKeypointIndex = 1;
inline constexpr int RightEyeKeypointIndex = 2;
inline constexpr int LeftEarKeypointIndex = 3;
inline constexpr int RightEarKeypointIndex = 4;
inline constexpr int LeftShoulderKeypointIndex = 5;
inline constexpr int RightShoulderKeypointIndex = 6;
inline constexpr int LeftElbowKeypointIndex = 7;
inline constexpr int RightElbowKeypointIndex = 8;
inline constexpr int LeftWristKeypointIndex = 9;
inline constexpr int RightWristKeypointIndex = 10;
inline constexpr int LeftHipKeypointIndex = 11;
inline constexpr int RightHipKeypointIndex = 12;
inline constexpr int LeftKneeKeypointIndex = 13;
inline constexpr int RightKneeKeypointIndex = 14;
inline constexpr int ShinKeypointsIndex = 25;

inline constexpr int minWidthFaceCrop = 130;
inline constexpr int minHeightFaceCrop = 190;
inline constexpr double minInterEyeDistance = 5.0;
inline constexpr float minBodyPtQuality = 0.1F;
inline constexpr int minNumberFrameAgeAndGender = 5;
inline constexpr int maxNumberFrameAgeAndGender = 200;

const std::vector< std::vector< int > > keyPointsLines = {
    {LeftShoulderKeypointIndex, RightShoulderKeypointIndex},
    {LeftElbowKeypointIndex, LeftShoulderKeypointIndex},
    {RightElbowKeypointIndex, RightShoulderKeypointIndex},
    {LeftWristKeypointIndex, LeftElbowKeypointIndex},
    {RightWristKeypointIndex, RightElbowKeypointIndex},
    {LeftHipKeypointIndex, LeftShoulderKeypointIndex},
    {RightHipKeypointIndex, RightShoulderKeypointIndex},
    {RightHipKeypointIndex, LeftHipKeypointIndex},
    {RightHipKeypointIndex, RightKneeKeypointIndex},
    {LeftHipKeypointIndex, LeftKneeKeypointIndex}};

/**
 * @brief Structure representing a crop region.
 */
struct CropRegion {
    int xmin{0};
    int xmax{0};
    int ymin{0};
    int ymax{0};

    /**
     * @brief Constructor for CropRegion.
     * @param _xmin Minimum x-coordinate.
     * @param _xmax Maximum x-coordinate.
     * @param _ymin Minimum y-coordinate.
     * @param _ymax Maximum y-coordinate.
     */
    inline CropRegion(int _xmin, int _xmax, int _ymin, int _ymax)
        : xmin(_xmin),
          xmax(_xmax),
          ymin(_ymin),
          ymax(_ymax){

          };

    /**
     * @brief Calculate the area of the crop region.
     * @return The area as an integer.
     */
    inline int area() const { return (xmax - xmin) * (ymax - ymin); };

    /**
     * @brief Check if the crop region is valid within given dimensions.
     * @param width The width of the bounding area.
     * @param height The height of the bounding area.
     * @return True if valid, false otherwise.
     */
    inline bool isvalid(int width, int height) const {
        return (xmin >= 0 && xmax <= width && ymin >= 0 && ymax <= height && area() > 0);
    };
};

/**
 * @brief Structure representing a body bounding box.
 */
struct BodyBBox {

    int x{0};
    int y{0};
    int width{0};
    int height{0};
    bool valid{false};
    int occupantid{0};

    /**
     * @brief Constructor for BodyBBox.
     * @param x_ X-coordinate of the bounding box.
     * @param y_ Y-coordinate of the bounding box.
     * @param w_ Width of the bounding box.
     * @param h_ Height of the bounding box.
     * @param valid_ Validity of the bounding box.
     * @param id Occupant ID.
     */
    BodyBBox(int x_, int y_, int w_, int h_, bool valid_, int id);

    /**
     * @brief Constructor for BodyBBox from a CSV reader.
     * @param reader The CSV reader.
     * @param frameindex The frame index.
     * @param id Occupant ID.
     */
    BodyBBox(const foms::CsvReader& reader, int frameindex, int id);
    /**
     * @brief Check if the bounding box is valid within given dimensions.
     * @param w The width of the bounding area.
     * @param h The height of the bounding area.
     * @return True if valid, false otherwise.
     */
    bool isvalid(int w, int h) const {
        return (x >= 0 && (x + width) <= w && y >= 0 && (y + height) <= h && width > 0 && height > 0);
    };

    /**
     * @brief Get the crop region for the bounding box.
     * @return The crop region.
     */
    CropRegion getCrop() const { return CropRegion(x, x + width, y, y + height); };
};

/**
 * @brief Structure representing a body keypoint.
 */
struct BodyKeyPoint {

    int x{0};
    int y{0};

    bool valid{false};
    int occupantid{0};
    float quality{0.0};

    /**
     * @brief Constructor for BodyKeyPoint.
     * @param x_ X-coordinate of the keypoint.
     * @param y_ Y-coordinate of the keypoint.
     * @param quality_ Quality of the keypoint.
     * @param id_ Occupant ID.
     */
    BodyKeyPoint(int x_, int y_, float quality_, int id_);

    /**
     * @brief Constructor for BodyKeyPoint from a CSV reader.
     * @param reader The CSV reader.
     * @param frameindex The frame index.
     * @param id Occupant ID.
     * @param keypointNumber The number of the keypoint.
     */
    BodyKeyPoint(const foms::CsvReader& reader, int frameindex, int id, int keypointNumber);
};

/**
 * @brief Enumeration for FOMS face error types.
 */
enum class FomsFaceErrorType {
    FomsFace_NOERROR = 0,          /**< No error */
    FomsFace_MISSINGKEYPOINT = 1,  /**< Missing keypoint */
    FomsFace_EYEINTERDISTANCE = 2, /**< Eye inter-distance error */
    FomsFace_SHINBADQUALITY = 3,   /**< Shin bad quality error */
    FomsFace_EARSBADQUALITY = 4,   /**< Ears bad quality error */
    FomsFace_BADCROPSIZE = 5,      /**< Bad crop error */
};

/**
 * @brief Compute the face crop from keypoints.
 * @param LkeyPt Left eye keypoint.
 * @param RkeyPt Right eye keypoint.
 * @param shinkeyPt Shin keypoint.
 * @param LEarkeyPt Left ear keypoint.
 * @param REarkeyPt Right ear keypoint.
 * @param faceCrop Output face crop region.
 * @return The face error type.
 */
FomsFaceErrorType computeFaceCropFromkeyPoints(const BodyKeyPoint& LkeyPt, const BodyKeyPoint& RkeyPt,
                                               const BodyKeyPoint& shinkeyPt, const BodyKeyPoint& LEarkeyPt,
                                               const BodyKeyPoint& REarkeyPt, foms::CropRegion& faceCrop);

/**
 * @brief Draw eyes, ears, and shin keypoints on an image.
 * @param img The image.
 * @param LkeyPt Left eye keypoint.
 * @param RkeyPt Right eye keypoint.
 * @param shinkeyPt Shin keypoint.
 * @param LEarkeyPt Left ear keypoint.
 * @param REarkeyPt Right ear keypoint.
 * @param color The color for drawing.
 */
void drawEyesEarsShin(cv::Mat& img, const BodyKeyPoint& LkeyPt, const BodyKeyPoint& RkeyPt,
                      const BodyKeyPoint& shinkeyPt, const BodyKeyPoint& LEarkeyPt,
                      const BodyKeyPoint& REarkeyPt, const cv::Scalar& color = cv::Scalar(0, 0, 255));

/**
 * @brief Draw lines between body keypoints on an image.
 * @param img The image.
 * @param bodyKeyPts The body keypoints.
 * @param color The color for drawing.
 * @param linesize The size of the lines.
 */
void drawLineKeyPts(cv::Mat& img, const std::vector< foms::BodyKeyPoint >& bodyKeyPts,
                    const cv::Scalar& color = cv::Scalar(0, 0, 255), int linesize = 4);

/**
 * @brief Constants for aggregating pose predictions and calculations.
 */

inline constexpr float poseDefaultModelThreshold = 0.5F;

struct cameraParameters {
    int imageHeight = 1984;
    int imageWidth = 2560;
    double distortionRadial2 = 0.33637919;
    double distortionRadial4 = -0.13904116;
    double distortionRadial6 = 0.01649714;
    double focalLengthX = 1062.22;
    double focalLengthY = 1062.22;
    double principalPointX = 1280;
    double principalPointY = 992;
    double distance_lateral_cameracenter_siegecenter_left = 1.0;
    double distance_lateral_cameracenter_siegecenter_right = 1.0;
    double distance_board_camera_long = 0.2;
};

cv::Mat getRotationMatrix(double roll, double pitch, double yaw);

foms::BodyKeyPoint transformBodyKeyPoint(const foms::BodyKeyPoint& keypoint, const cv::Mat& H);

} // namespace foms

#endif // FOMS_UTILS_HPP