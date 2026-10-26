cmake_minimum_required (VERSION 3.8)
project (EXACT)

enable_testing()

SET (PLATFORM 64)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -stdlib=libstdc++")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -fsanitize=address -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -g -Wall -O1 -funroll-loops -msse3 -fsanitize=address -fno-omit-frame-pointer -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -g -Wall -O1 -funroll-loops -msse3 -fsanitize=address -fno-omit-frame-pointer -DNAN_CHECKS -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -DNAN_CHECKS")

# 1 This line for local
#SET (CMAKE_CXX_FLAGS                "-Wall -O3 -funroll-loops  -msse3 -fsanitize=address -fno-omit-frame-pointer -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=gnu++17 -Wall -O3 -funroll-loops  -msse3 -fsanitize=address -fno-omit-frame-pointer -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++20 -Wall -O3 -funroll-loops  -msse3 -fsanitize=address -fno-omit-frame-pointer -D_GLIBCXX_DEBUG")

# 2 This line for cluster
#SET (CMAKE_CXX_FLAGS                "-std=gnu++17 -Wall -O3 -funroll-loops  -msse3 -fno-omit-frame-pointer -D_GLIBCXX_DEBUG")

SET (CMAKE_CXX_FLAGS                " -Wall -O3 -funroll-loops ")
#SET (CMAKE_CXX_FLAGS                " -Wall -O1 -funroll-loops -msse3 -g -fsanitize=address -fno-omit-frame-pointer -shared-libasan -DGLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS} -g")
#SET (CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS} -Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS} -O4 -funroll-loops -DNDEBUG")

#SET (CMAKE_SHARED_LINKER_FLAGS      " -Wall -O1 -funroll-loops -msse3 -g -fsanitize=address -fno-omit-frame-pointer -shared-libasan -DGLIBCXX_DEBUG")


set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

message(STATUS "${CMAKE_CXX_FLAGS}")
message(STATUS "project source dir is ${PROJECT_SOURCE_DIR}")

find_package(MPI)
#SET(MPI_INCLUDE_PATH /usr/local/Cellar/open-mpi/3.1.2/include)
#SET(MPI_INCLUDE_PATH /usr/local/Cellar/open-mpi/4.0.0/include)
#SET(MPI_INCLUDE_PATH /usr/local/Cellar/open-mpi/4.0.1/include)
SET(MPI_INCLUDE_PATH /opt/homebrew/Cellar/open-mpi/5.0.9/include)
#SET(MPI_LIBRARY /usr/local/Cellar/open-mpi/3.1.2/lib/libmpi.dylib)
#SET(MPI_LIBRARY /usr/local/Cellar/open-mpi/4.0.0/lib/libmpi.dylib)
#SET(MPI_LIBRARY /usr/local/Cellar/open-mpi/4.0.1/lib/libmpi.dylib)
#SET(MPI_LIBRARY /usr/local/Cellar/open-mpi/5.0.1/lib/libmpi.dylib)
SET(MPI_LIBRARY /opt/homebrew/Cellar/open-mpi/5.0.9/lib/libmpi.dylib)
MESSAGE(STATUS "MPI include directory: ${MPI_INCLUDE_PATH}")
MESSAGE(STATUS "MPI library: ${MPI_LIBRARY}")
MESSAGE(STATUS "MPI extra: ${MPI_EXTRA}")
include_directories(${MPI_INCLUDE_PATH})

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(common)
add_subdirectory(time_series)

add_subdirectory(rnn)
add_subdirectory(rnn_tests)
add_subdirectory(rnn_examples)

add_subdirectory(weights)
add_subdirectory(examm)

add_subdirectory(mpi)

# add_subdirectory(exona)
