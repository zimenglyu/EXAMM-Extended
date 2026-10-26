#ifndef EXAMM_COMMON_FILES_HXX
#define EXAMM_COMMON_FILES_HXX

#include <stdexcept>
using std::runtime_error;

#include <string>
using std::string;

string get_file_as_string(string file_path) noexcept(false);

int mkpath(const char* path, mode_t mode);

#endif
