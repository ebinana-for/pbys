#include "foms_inference.hpp"
#include <algorithm>
#include <vector>

foms::InferenceObject::InferenceObject(std::string_view modelfile)
    : m_modelfile{modelfile}

{}

std::pair< size_t, float > foms::get_max_element(const float* values, size_t size) {
    const std::vector< float > result_vec(values, values + size);
    const size_t max_idx = std::max_element(result_vec.begin(), result_vec.end()) - result_vec.begin();
    return std::make_pair(max_idx, result_vec[max_idx]);
}

double foms::getexetime() {

    timeval thisTime = {0, 0};
    long seconds(0);
    long useconds(0);

    gettimeofday(&thisTime, nullptr);

    seconds = thisTime.tv_sec;
    useconds = thisTime.tv_usec;

    return (static_cast< double >(seconds) * 1000000.0 + static_cast< double >(useconds));
}
