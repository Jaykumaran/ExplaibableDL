CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)
CMAKE_POLICY(VERSION 3.5.1)

GET_FILENAME_COMPONENT(PROJECT_NAME ${CMAKE_SOURCE_DIR} NAME)
PROJECT(${PROJECT_NAME})

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
CMAKE_POLICY(SET CMP0069 NEW) 
SET(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
SET(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
ENDIF()
IF(NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-newline-eof")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reserved-id-macro")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-disabled-macro-expansion")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-exit-time-destructors")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
ENDIF()

FILE(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/source/*")
ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/source/")

SET(THREADS_PREFER_PTHREAD_FLAG ON)
FIND_PACKAGE(Threads REQUIRED)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE Threads::Threads)

FIND_PACKAGE(OpenCV REQUIRED)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

FIND_PACKAGE(glfw3 3.3 REQUIRED)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE "")
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE glfw)

SET(OpenGL_GL_PREFERENCE "GLVND")
FIND_PACKAGE(OpenGL REQUIRED)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${OPENGL_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OPENGL_LIBRARIES})

LIST(APPEND CMAKE_MODULE_PATH "/dependencies/g2o/cmake_modules")
FIND_PACKAGE(G2O REQUIRED)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${G2O_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} g2o_solver_eigen g2o_types_sba g2o_types_slam3d)

FIND_PACKAGE(Eigen3 3.3 REQUIRED NO_MODULE)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
