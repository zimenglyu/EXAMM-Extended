find_package(MPI)

if (MPI_FOUND)
    set (CMAKE_CXX_COMPILE_FLAGS "${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})

    add_executable(examm_mpi examm_mpi.cxx)
    target_link_libraries(examm_mpi examm_strategy exact_time_series exact_common exact_weights examm_nn ${MPI_LIBRARIES} ${MPI_EXTRA} pthread)
endif (MPI_FOUND)
