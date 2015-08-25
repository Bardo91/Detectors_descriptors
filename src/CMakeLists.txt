cmake_minimum_required(VERSION 2.4)
project(DetectorDescriptors)


# Collect all sources
file(GLOB_RECURSE BAGPFWORDS_SOURCE_FILES "./code/*.cpp" "./code/*.h" "./code/*.inl")

# Clasify sources according to folder structure.
# This creates a nice layout for the project in visual studio
# This macro is courtesy of http://www.cmake.org/pipermail/cmake/2013-November/056336.html
macro(GroupSources curdir)
	file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir}
		${PROJECT_SOURCE_DIR}/${curdir}/*)
	foreach(child ${children})
		if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			GroupSources(${curdir}/${child})
		else()
			string(REPLACE "/" "\\" groupname ${curdir})
			source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
		endif()
	endforeach()
endmacro()

GroupSources(src)

if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-std=c++11)
endif()

##set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_LIB_DIR})
message(STATUS ${OpenCV_LIBS})

add_executable(DetectorDescriptors ${BAGPFWORDS_SOURCE_FILES})
target_include_directories(DetectorDescriptors PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(DetectorDescriptors PUBLIC ${PROJECT_SOURCE_DIR}/../lib/bovil/src/code)
target_link_libraries(DetectorDescriptors LINK_PUBLIC BOViL)
target_link_libraries(DetectorDescriptors LINK_PUBLIC ${OpenCV_LIBS})

