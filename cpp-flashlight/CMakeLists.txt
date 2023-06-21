cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ReinforceCartPole ReinforceCartPole.cpp CartPoleEnv.cpp ChooseAlong.cpp Join.cpp)

find_package(flashlight CONFIG REQUIRED)
target_link_libraries(ReinforceCartPole PRIVATE flashlight::flashlight)
