#ifndef EXAMM_PI_TRAINING_DATA_HXX
#define EXAMM_PI_TRAINING_DATA_HXX

#include <string>
#include <vector>

#include "common/arguments.hxx"

namespace pi_training {

inline std::vector<std::string> dataset_filenames(const std::vector<std::string>& arguments) {
    std::vector<std::string> files;
    if (argument_exists(arguments, "--filenames")) {
        get_argument_vector(arguments, "--filenames", true, files);
    } else {
        get_argument_vector(arguments, "--training_filenames", true, files);
        get_argument_vector(arguments, "--validation_filenames", true, files);
    }
    return files;
}

}  // namespace pi_training

#endif
