# This code is from the CMake FAQ

if (NOT EXISTS "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: \"/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/install_manifest.txt\"")
endif(NOT EXISTS "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/install_manifest.txt")

file(READ "/home/yxh/文档/yolov5/Open3D/3rdparty/turbojpeg/src/ext_turbojpeg-build/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
list(REVERSE files)
foreach (file ${files})
  message(STATUS "Uninstalling \"$ENV{DESTDIR}${file}\"")
    if (EXISTS "$ENV{DESTDIR}${file}")
      execute_process(
        COMMAND "/home/yxh/anaconda3/lib/python3.6/site-packages/cmake/data/bin/cmake" -E remove "$ENV{DESTDIR}${file}"
        OUTPUT_VARIABLE rm_out
        RESULT_VARIABLE rm_retval
      )
    if(NOT ${rm_retval} EQUAL 0)
      message(FATAL_ERROR "Problem when removing \"$ENV{DESTDIR}${file}\"")
    endif (NOT ${rm_retval} EQUAL 0)
  else (EXISTS "$ENV{DESTDIR}${file}")
    message(STATUS "File \"$ENV{DESTDIR}${file}\" does not exist.")
  endif (EXISTS "$ENV{DESTDIR}${file}")
endforeach(file)
