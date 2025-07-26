# Install script for directory: /home/yxh/文档/yolov5/Open3D/src/Open3D

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/yxh/文档/yolov5")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/yxh/文档/yolov5/include/Open3D")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/yxh/文档/yolov5/include" TYPE DIRECTORY FILES "/home/yxh/文档/yolov5/Open3D/src/Open3D" REGEX "/Visualization\\/Shader\\/GLSL$" EXCLUDE REGEX "/[^/]*\\.cpp$" EXCLUDE REGEX "/[^/]*\\.in$" EXCLUDE REGEX "/[^/]*\\.txt$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/yxh/文档/yolov5/lib/libOpen3D.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/yxh/文档/yolov5/lib/libOpen3D.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/yxh/文档/yolov5/lib/libOpen3D.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/yxh/文档/yolov5/lib/libOpen3D.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/yxh/文档/yolov5/lib" TYPE SHARED_LIBRARY FILES "/home/yxh/文档/yolov5/Open3D/lib/libOpen3D.so")
  if(EXISTS "$ENV{DESTDIR}/home/yxh/文档/yolov5/lib/libOpen3D.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/yxh/文档/yolov5/lib/libOpen3D.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/yxh/文档/yolov5/lib/libOpen3D.so"
         OLD_RPATH "/home/yxh/文档/yolov5/Open3D/Open3D:/home/yxh/文档/yolov5/Open3D/3rdparty_install/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/yxh/文档/yolov5/lib/libOpen3D.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/yxh/文档/yolov5/include/Open3D/Open3D.h;/home/yxh/文档/yolov5/include/Open3D/Open3DConfig.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/yxh/文档/yolov5/include/Open3D" TYPE FILE FILES
    "/home/yxh/文档/yolov5/Open3D/src/Open3D/Open3D.h"
    "/home/yxh/文档/yolov5/Open3D/src/Open3D/Open3DConfig.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Open3D" TYPE FILE FILES
    "/home/yxh/文档/yolov5/Open3D/CMakeFiles/Open3DConfig.cmake"
    "/home/yxh/文档/yolov5/Open3D/Open3DConfigVersion.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/Camera/cmake_install.cmake")
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/ColorMap/cmake_install.cmake")
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/Geometry/cmake_install.cmake")
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/Integration/cmake_install.cmake")
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/Odometry/cmake_install.cmake")
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/Registration/cmake_install.cmake")
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/Utility/cmake_install.cmake")
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/IO/cmake_install.cmake")
  include("/home/yxh/文档/yolov5/Open3D/src/Open3D/Visualization/cmake_install.cmake")

endif()

