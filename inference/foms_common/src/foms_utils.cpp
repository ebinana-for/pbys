
#include "foms_utils.hpp"
#include <cmath>

foms::FomsFaceErrorType
foms::computeFaceCropFromkeyPoints(const foms::BodyKeyPoint& LkeyPt, const foms::BodyKeyPoint& RkeyPt,
                                   const foms::BodyKeyPoint& shinkeyPt, const foms::BodyKeyPoint& LEarkeyPt,
                                   const foms::BodyKeyPoint& REarkeyPt, foms::CropRegion& faceCrop) {

    if (!LkeyPt.valid || !RkeyPt.valid || !shinkeyPt.valid || !REarkeyPt.valid || !LEarkeyPt.valid) {
        return foms::FomsFaceErrorType::FomsFace_MISSINGKEYPOINT;
    }

    const double interEyeDistance = std::sqrt((LkeyPt.x - RkeyPt.x) * (LkeyPt.x - RkeyPt.x) +
                                              (LkeyPt.y - RkeyPt.y) * (LkeyPt.y - RkeyPt.y));

    if (interEyeDistance < minInterEyeDistance) {
        return foms::FomsFaceErrorType::FomsFace_EYEINTERDISTANCE;
    }

    if (shinkeyPt.x < LkeyPt.x || shinkeyPt.x > RkeyPt.x) {
        return foms::FomsFaceErrorType::FomsFace_SHINBADQUALITY;
    }

    if (LEarkeyPt.x > RkeyPt.x && REarkeyPt.x > RkeyPt.x) {
        return foms::FomsFaceErrorType::FomsFace_EARSBADQUALITY;
    }
    if (LEarkeyPt.x < LkeyPt.x && REarkeyPt.x < LkeyPt.x) {
        return foms::FomsFaceErrorType::FomsFace_EARSBADQUALITY;
    }

    const double eye_to_chin_vert_distance = (shinkeyPt.y - (RkeyPt.y + LkeyPt.y) / 2.0);

    if (eye_to_chin_vert_distance <= interEyeDistance * 1.5) {
        return foms::FomsFaceErrorType::FomsFace_SHINBADQUALITY;
    }

    const double center_x = (REarkeyPt.x + LEarkeyPt.x) / 2.0;
    const double center_y = (LkeyPt.y + RkeyPt.y) / 2.0;
    double width = interEyeDistance * 3.2;
    double height = eye_to_chin_vert_distance * 2.2;

    faceCrop.xmin = static_cast< int >(lround(center_x - width / 2.0 + 0.5));
    faceCrop.ymin = static_cast< int >(lround(center_y - 0.8 * height / 2.0 + 0.5));

    faceCrop.xmax = static_cast< int >(lround(center_x + width / 2.0 + 0.5));
    faceCrop.ymax = static_cast< int >(lround(center_y + height / 2.0 + 0.5));

    width = static_cast< double >(faceCrop.xmax - faceCrop.xmin);
    height = static_cast< double >(faceCrop.ymax - faceCrop.ymin);

    // std::cout << "Face crop region: " << width << "x" << height << " " << width / height << std::endl;

    const double ratio = width / height;

    if (ratio < 0.6 || ratio > 2.0 || width < minWidthFaceCrop || height < minHeightFaceCrop) {
        return foms::FomsFaceErrorType::FomsFace_BADCROPSIZE;
    }

    return foms::FomsFaceErrorType::FomsFace_NOERROR;
};

void foms::drawEyesEarsShin(cv::Mat& img, const BodyKeyPoint& LkeyPt, const BodyKeyPoint& RkeyPt,
                            const BodyKeyPoint& shinkeyPt, const BodyKeyPoint& LEarkeyPt,
                            const BodyKeyPoint& REarkeyPt, const cv::Scalar& color) {

    if (LkeyPt.valid) {
        cv::circle(img, cv::Point(LkeyPt.x, LkeyPt.y), 4, color, 1);
    }
    if (RkeyPt.valid) {
        cv::circle(img, cv::Point(RkeyPt.x, RkeyPt.y), 4, cv::Scalar(0, 255, 0), 1);
    }
    if (shinkeyPt.valid) {
        cv::circle(img, cv::Point(shinkeyPt.x, shinkeyPt.y), 2, color, -1);
    }
    if (LEarkeyPt.valid) {
        cv::circle(img, cv::Point(LEarkeyPt.x, LEarkeyPt.y), 2, color, -1);
    }
    if (REarkeyPt.valid) {
        cv::circle(img, cv::Point(REarkeyPt.x, REarkeyPt.y), 2, color, -1);
    }
};

void foms::drawLineKeyPts(cv::Mat& img, const std::vector< foms::BodyKeyPoint >& bodyKeyPts,
                          const cv::Scalar& color, int linesize) {

    for (const auto& index_values : foms::keyPointsLines) {

        const foms::BodyKeyPoint& p1 = bodyKeyPts[index_values[0]];
        const foms::BodyKeyPoint& p2 = bodyKeyPts[index_values[1]];

        if (p1.valid && p2.valid) {
            cv::line(img, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), color, linesize);
        }
    }
}

foms::BodyBBox::BodyBBox(int x_, int y_, int w_, int h_, bool valid_, int id)
    : x(x_),
      y(y_),
      width(w_),
      height(h_),
      valid(valid_),
      occupantid(id){

      };
foms::BodyBBox::BodyBBox(const foms::CsvReader& reader, int frameindex, int id) : occupantid(id) {

    const std::string qualityName = getBBoxValidFieldName(occupantid);
    const std::string qualityString = reader.getValueByName(qualityName, frameindex);

    valid = (qualityString == "true");
    if (!valid) {
        return;
    }
    try {
        const std::string xName = getBBoxXFieldName(occupantid);
        const std::string xString = reader.getValueByName(xName, frameindex);
        x = std::stoi(xString);

        const std::string yName = getBBoxYFieldName(occupantid);
        const std::string yString = reader.getValueByName(yName, frameindex);
        y = std::stoi(yString);

        const std::string wName = getBBoxWFieldName(occupantid);
        const std::string wString = reader.getValueByName(wName, frameindex);
        width = std::stoi(wString);

        const std::string hName = getBBoxHFieldName(occupantid);
        const std::string hString = reader.getValueByName(hName, frameindex);
        height = std::stoi(hString);

    } catch (const std::invalid_argument& ia) {
        g_logger.writeLog(ia.what(), foms::errorLevel::INFO);
        valid = false;
    }
};

foms::BodyKeyPoint::BodyKeyPoint(int x_, int y_, float quality_, int id_)
    : x(x_),
      y(y_),
      occupantid(id_),
      quality(quality_)

{
    valid = (quality > minBodyPtQuality);
};

foms::BodyKeyPoint::BodyKeyPoint(const foms::CsvReader& reader, int frameindex, int id, int keypointNumber)
    : occupantid(id) {

    const std::string qualityName = getBodykeypointQualityFieldName(occupantid, keypointNumber);
    const std::string qualityString = reader.getValueByName(qualityName, frameindex);

    try {
        quality = std::stof(qualityString);
        valid = (quality > minBodyPtQuality);
        if (!valid) {
            return;
        }
        const std::string xName = getBodykeypointXFieldName(occupantid, keypointNumber);
        const std::string xString = reader.getValueByName(xName, frameindex);
        x = std::stoi(xString);

        const std::string yName = getBodykeypointYFieldName(occupantid, keypointNumber);
        const std::string yString = reader.getValueByName(yName, frameindex);
        y = std::stoi(yString);

    } catch (const std::invalid_argument& ia) {
        g_logger.writeLog(ia.what(), foms::errorLevel::INFO);
        valid = false;
    }
};

cv::Mat foms::getRotationMatrix(double roll, double pitch, double yaw) {

    // Create rotation matrices for each axis
    // Roll (rotation around x-axis), Pitch (rotation around y-axis), Yaw (rotation around z-axis)
    const cv::Mat R_x =
        (cv::Mat_< double >(3, 3) << 1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll));
    const cv::Mat R_y =
        (cv::Mat_< double >(3, 3) << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch));
    const cv::Mat R_z = (cv::Mat_< double >(3, 3) << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);

    const cv::Mat R = R_z * R_y * R_x;

    // Rc is the rotation matrix for the camera coordinate system
    // It is defined such that it transforms points from the camera coordinate system to the world coordinate
    cv::Mat Rc = cv::Mat::zeros(3, 3, CV_64F);

    Rc.at< double >(0, 2) = 1.0;
    Rc.at< double >(1, 0) = 1.0;
    Rc.at< double >(2, 1) = -1.0;

    // The final rotation matrix is obtained by multiplying the camera rotation matrix with the rotation
    // matrix for the specified roll, pitch, and yaw angles.
    return Rc.inv() * R * Rc;
}

foms::BodyKeyPoint foms::transformBodyKeyPoint(const foms::BodyKeyPoint& keypoint, const cv::Mat& H) {

    foms::BodyKeyPoint transformedKeypoint = keypoint;

    if (keypoint.valid) {
        const cv::Mat point = (cv::Mat_< double >(3, 1) << keypoint.x, keypoint.y, 1.0);
        cv::Mat transformedPoint = H * point;

        if (transformedPoint.at< double >(2, 0) != 0) {
            transformedKeypoint.x =
                static_cast< int >(transformedPoint.at< double >(0, 0) / transformedPoint.at< double >(2, 0));
            transformedKeypoint.y =
                static_cast< int >(transformedPoint.at< double >(1, 0) / transformedPoint.at< double >(2, 0));
        }
    }
    return transformedKeypoint;
}