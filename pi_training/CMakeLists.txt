file(GLOB_RECURSE pi_shared_sources CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/common/*.cxx" "${PROJECT_SOURCE_DIR}/common/*.hxx"
    "${PROJECT_SOURCE_DIR}/examm/*.cxx" "${PROJECT_SOURCE_DIR}/examm/*.hxx"
    "${PROJECT_SOURCE_DIR}/rnn/*.cxx" "${PROJECT_SOURCE_DIR}/rnn/*.hxx"
    "${PROJECT_SOURCE_DIR}/time_series/*.cxx" "${PROJECT_SOURCE_DIR}/time_series/*.hxx"
    "${PROJECT_SOURCE_DIR}/weights/*.cxx" "${PROJECT_SOURCE_DIR}/weights/*.hxx")
list(APPEND pi_shared_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/pi_bridge.hxx"
    "${CMAKE_CURRENT_SOURCE_DIR}/pi_data.hxx"
    "${CMAKE_CURRENT_SOURCE_DIR}/pi_protocol.hxx"
    "${CMAKE_CURRENT_SOURCE_DIR}/pi_train_server.cxx")
list(SORT pi_shared_sources)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${pi_shared_sources})
set(pi_source_manifest "")
foreach(source IN LISTS pi_shared_sources)
    file(RELATIVE_PATH source_name "${PROJECT_SOURCE_DIR}" "${source}")
    file(SHA256 "${source}" source_hash)
    string(APPEND pi_source_manifest "${source_name}:${source_hash}\n")
endforeach()
string(SHA256 pi_source_hash "${pi_source_manifest}")

add_executable(pi_train_server pi_train_server.cxx)
target_compile_definitions(pi_train_server PRIVATE EXAMM_PI_SOURCE_FINGERPRINT="${pi_source_hash}")
target_link_libraries(pi_train_server PRIVATE examm_strategy exact_time_series exact_common exact_weights examm_nn pthread)
if(TARGET examm_mpi)
    target_compile_definitions(examm_mpi PRIVATE EXAMM_PI_SOURCE_FINGERPRINT="${pi_source_hash}")
endif()
