add_executable(math_test exp)
target_compile_definitions(math_test PUBLIC -DEXACT_MATH_TEST)

add_library(exact_common arguments.cxx exp.cxx random.cxx color_table.cxx log.cxx files.cxx process_arguments.cxx)
target_link_libraries(exact_common examm_strategy exact_time_series)
