cmake_minimum_required (VERSION 2.6)
project (YCDL)

# config gloal output
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/output)

# c++ 11
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-pthread -std=c++11 -g")
include_directories (
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/thirdparty/ps-lite/include"
    "${PROJECT_SOURCE_DIR}/thirdparty/eigen-3.3.7"
    "${PROJECT_SOURCE_DIR}/thirdparty/json/include"
)
link_directories(
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_BINARY_DIR}/thirdparty/ps-lite"
)

add_subdirectory(thirdparty/ps-lite)

set(STA_DEPS pslite protobuf zmq)

# add src executable
file(GLOB example_sources "src/*.cpp")
foreach( example_source_file ${example_sources} )
  get_filename_component(filename ${example_source_file} NAME)
  string(REPLACE ".cpp" "" example_name ${filename} )
  add_executable(${example_name} ${example_source_file})
  target_link_libraries(${example_name} ${STA_DEPS})
  set_target_properties(${example_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
          ${PROJECT_SOURCE_DIR}/bin/output )
endforeach()

# add test executable
file(GLOB example_sources "test/*.cpp")
foreach( example_source_file ${example_sources} )
  get_filename_component(filename ${example_source_file} NAME)
  string(REPLACE ".cpp" "" example_name ${filename} )
  add_executable(${example_name} ${example_source_file})
  target_link_libraries(${example_name} ${STA_DEPS})
  set_target_properties(${example_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
      ${PROJECT_SOURCE_DIR}/bin/test)
endforeach()
