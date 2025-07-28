cmake_minimum_required(VERSION 3.10)
project(base LANGUAGES CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) 找 OpenCV
find_package(OpenCV REQUIRED)

# 2) 手动指定 MVS 路径（安装默认 /opt/MVS）
set(MVS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/hikcamera/SDK")
set(MVS_INCLUDE_DIR "${MVS_ROOT}/include")
set(MVS_LIB_DIR     "${MVS_ROOT}/lib/64")

# 3) 添加可执行文件
add_executable(${PROJECT_NAME}
    main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hikcamera/src/HikCamera.cpp
)

# 4) 包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${MVS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/hikcamera/include
)

# 5) 链接库
target_link_directories(${PROJECT_NAME} PRIVATE ${MVS_LIB_DIR})
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    MvCameraControl
)

# 6) 运行时 rpath，防止找不到 .so
target_link_options(${PROJECT_NAME} PRIVATE
    "-Wl,-rpath,${MVS_LIB_DIR}"
)

# 安装/复制配置文件到可执行文件同目录
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/camera_set.json
    ${CMAKE_CURRENT_BINARY_DIR}/camera_set.json
    COPYONLY
)