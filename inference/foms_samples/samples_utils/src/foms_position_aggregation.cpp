#include "foms_position_aggregation.hpp"

void foms::PredictionAggregator::aggregate_predictions() {
    // Implement all individual logics of aggregating models, output over time and with keypoints
    aggregate_closetoairbag();
    aggregate_pelvisrotation();
    aggregate_torsodissimetry();
    aggregate_oneshoulderincontact();
    aggregate_pelvisdrift();
    aggregate_legsondashboard();
    aggregate_reachingback();
    aggregate_lateralbodycontact();

    current_output.valid = true;
    current_output.scores.valid = true;
}

void foms::PredictionAggregator::update_observation(const foms::positionsScores& model_scores,
                                                    const foms::positionsMeasures& measures) {
    // Implement all individual logics of aggregating models, output over time and with keypoints
    closetoairbagMLBuffer.push(model_scores.closetoairbag);
    pelvisrotationMLBuffer.push(model_scores.pelvisrotation);
    torsodissimetryMLBuffer.push(model_scores.torsodissimetry);
    oneshoulderincontactMLBuffer.push(model_scores.oneshoulderincontact);
    pelvisdriftMLBuffer.push(model_scores.pelvisdrift);
    legsondashboardMLBuffer.push(model_scores.legsondashboard);
    reachingbackMLBuffer.push(model_scores.reachingback);
    lateralbodycontactMLBuffer.push(model_scores.lateralbodycontact);

    dashboardDistanceBuffer.push(measures.dashboardDistance);
    legsOrientationBuffer.push(measures.legsOrientation);
    shoulderOrientationBuffer.push(measures.shoulderOrientation);

    aggregate_predictions();
}

void foms::PredictionAggregator::aggregate_closetoairbag() {

    const std::array< float, poseDefaultBufferSize > distances = dashboardDistanceBuffer.get_buffer();
    const bool modelClose = closetoairbagMLBuffer.get_mean_value() > poseDefaultModelThreshold;

    unsigned int sumCloseness = 0;
    for (const auto d : distances) {
        sumCloseness += static_cast< unsigned int >(d < distanceDashboardThreshold);
    }

    current_output.closetoairbag = modelClose && sumCloseness > 3 * poseDefaultBufferSize / 4;
    current_output.scores.closetoairbag = closetoairbagMLBuffer.get_mean_value();
}

void foms::PredictionAggregator::aggregate_pelvisrotation() {
    current_output.scores.pelvisrotation = pelvisrotationMLBuffer.get_mean_value();
    current_output.pelvisrotation =
        current_output.scores.pelvisrotation > poseDefaultModelThreshold && pelvisrotationMLBuffer.filled();
}
void foms::PredictionAggregator::aggregate_torsodissimetry() {

    const std::array< float, poseDefaultBufferSize > orientations = shoulderOrientationBuffer.get_buffer();
    current_output.scores.torsodissimetry = torsodissimetryMLBuffer.get_mean_value();

    const bool modelDissimetry = torsodissimetryMLBuffer.get_mean_value() > poseDefaultModelThreshold;

    unsigned int sumDissimetry = 0;
    for (const auto d : orientations) {
        sumDissimetry += static_cast< unsigned int >(d > orientationShoulderThreshold * M_PI / 180.0F);
    }

    current_output.torsodissimetry = modelDissimetry && sumDissimetry > 3 * poseDefaultBufferSize / 4;
}

void foms::PredictionAggregator::aggregate_oneshoulderincontact() {
    current_output.scores.oneshoulderincontact = oneshoulderincontactMLBuffer.get_mean_value();
    current_output.oneshoulderincontact =
        current_output.scores.oneshoulderincontact > poseDefaultModelThreshold &&
        oneshoulderincontactMLBuffer.filled();
}

void foms::PredictionAggregator::aggregate_pelvisdrift() {
    current_output.scores.pelvisdrift = pelvisdriftMLBuffer.get_mean_value();
    current_output.pelvisdrift =
        current_output.scores.pelvisdrift > poseDefaultModelThreshold && pelvisdriftMLBuffer.filled();
}
void foms::PredictionAggregator::aggregate_legsondashboard() {
    current_output.scores.legsondashboard = legsondashboardMLBuffer.get_mean_value();
    current_output.legsondashboard =
        current_output.scores.legsondashboard > poseDefaultModelThreshold && legsondashboardMLBuffer.filled();
}
void foms::PredictionAggregator::aggregate_reachingback() {
    current_output.scores.reachingback = reachingbackMLBuffer.get_mean_value();
    current_output.reachingback =
        current_output.scores.reachingback > poseDefaultModelThreshold && reachingbackMLBuffer.filled();
}
void foms::PredictionAggregator::aggregate_lateralbodycontact() {
    current_output.scores.lateralbodycontact = lateralbodycontactMLBuffer.get_mean_value();
    current_output.lateralbodycontact =
        current_output.scores.lateralbodycontact > poseDefaultModelThreshold &&
        lateralbodycontactMLBuffer.filled();
}

foms::positionsScores foms::PredictionAggregator::get_framewise_model_scores() const {
    foms::positionsScores model_confidence{};
    model_confidence.valid = true;
    model_confidence.closetoairbag = closetoairbagMLBuffer.get_head_value();
    model_confidence.pelvisdrift = pelvisdriftMLBuffer.get_head_value();
    model_confidence.pelvisrotation = pelvisrotationMLBuffer.get_head_value();
    model_confidence.torsodissimetry = torsodissimetryMLBuffer.get_head_value();
    model_confidence.oneshoulderincontact = oneshoulderincontactMLBuffer.get_head_value();
    model_confidence.legsondashboard = legsondashboardMLBuffer.get_head_value();
    model_confidence.reachingback = reachingbackMLBuffer.get_head_value();
    model_confidence.lateralbodycontact = lateralbodycontactMLBuffer.get_head_value();
    return model_confidence;
}
