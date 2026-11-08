cmake_minimum_required(VERSION 3.18)
project(eopy_native LANGUAGES CXX)

add_library(eopykernels SHARED
    src/eopy_native/eopy_kernels.cpp
)

# Read the same literal used by Python and scikit-build-core's metadata provider.
# Reconfigure when it changes, including during an incremental native build.
set(EOPY_VERSION_FILE "${PROJECT_SOURCE_DIR}/eopy/version.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${EOPY_VERSION_FILE}")
file(STRINGS "${EOPY_VERSION_FILE}" EOPY_VERSION_LINE REGEX "^__version__ = ")
string(REGEX REPLACE "^__version__ = \"([^\"]+)\"$" "\\1" EOPY_VERSION "${EOPY_VERSION_LINE}")
target_compile_definitions(eopykernels PRIVATE EOPY_VERSION="${EOPY_VERSION}")

target_compile_features(eopykernels PRIVATE cxx_std_17)
target_include_directories(eopykernels PUBLIC src/eopy_native)
set_target_properties(eopykernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(NOT SKBUILD)
    set_target_properties(eopykernels PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/eopy"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/eopy")
endif()

install(TARGETS eopykernels
    LIBRARY DESTINATION eopy
    RUNTIME DESTINATION eopy
    ARCHIVE DESTINATION eopy
)
