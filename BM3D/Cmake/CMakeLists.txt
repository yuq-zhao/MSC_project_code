cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)

project(bm3d)

find_package(OpenCV REQUIRED)


file(GLOB srcfiles ./*.cpp)

add_executable(bm3d ${srcfiles})

target_include_directories(bm3d PRIVATE
${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(bm3d PRIVATE
${OpenCV_LIBS}
)

if(NOT ((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR MSVC))
find_package(Boost REQUIRED)
target_include_directories(bm3d PRIVATE
${Boost_INCLUDE_DIRS}
)
endif()