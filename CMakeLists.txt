cmake_minimum_required(VERSION 3.18)
project(eopy_native LANGUAGES CXX)

add_library(eopykernels SHARED
    src/eopy_native/eopy_kernels.cpp
)

target_compile_features(eopykernels PRIVATE cxx_std_17)
target_include_directories(eopykernels PUBLIC src/eopy_native)
set_target_properties(eopykernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/eopy"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/eopy"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

install(TARGETS eopykernels
    LIBRARY DESTINATION eopy
    RUNTIME DESTINATION eopy
    ARCHIVE DESTINATION eopy
)
