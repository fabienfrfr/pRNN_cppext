cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pRNN)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pRNN pRNN.cpp)
target_link_libraries(pRNN "${TORCH_LIBRARIES}")
set_property(TARGET pRNN PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Unix.

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pRNN
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:pRNN>)
endif (MSVC)
