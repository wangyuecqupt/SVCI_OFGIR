# Install script for directory: /home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/yxh/文档/yolov5/Open3D/3rdparty_install")
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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/libturbojpeg.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "tjbench" FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/tjbench-static")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/turbojpeg.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/libjpeg.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "cjpeg" FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/cjpeg-static")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "djpeg" FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/djpeg-static")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "jpegtran" FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/jpegtran-static")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/rdjpgcom")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/wrjpgcom")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/libjpeg-turbo" TYPE FILE FILES
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/README.ijg"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/README.md"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/example.txt"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/tjexample.c"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/libjpeg.txt"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/structure.txt"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/usage.txt"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/wizard.txt"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/LICENSE.md"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1" TYPE FILE FILES
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/cjpeg.1"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/djpeg.1"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/jpegtran.1"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/rdjpgcom.1"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/wrjpgcom.1"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES
    "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/pkgscripts/libjpeg.pc"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/pkgscripts/libturbojpeg.pc"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/jconfig.h"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/jerror.h"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/jmorecfg.h"
    "/home/yxh/文档/yolov5/Open3D/3rdparty/libjpeg-turbo/libjpeg-turbo/jpeglib.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/md5/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
