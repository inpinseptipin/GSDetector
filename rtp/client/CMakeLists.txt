#
#			AuxPort Library
#			"Generic Modules to facilitate C++ Windows Application development" - inpinseptipin
#
#			BSD 3-Clause License
#
#			Copyright (c) 2020, Satyarth Arora
#			All rights reserved.
#
#			Redistribution and use in source and binary forms, with or without
#			modification, are permitted provided that the following conditions are met:
#
#			1. Redistributions of source code must retain the above copyright notice, this
#			   list of conditions and the following disclaimer.
#
#			2. Redistributions in binary form must reproduce the above copyright notice,
#			   this list of conditions and the following disclaimer in the documentation
#			   and/or other materials provided with the distribution.
#
#			3. Neither the name of the copyright holder nor the names of its
#			   contributors may be used to endorse or promote products derived from
#			   this software without specific prior written permission.
#
#			THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#			AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#			IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#			DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#			FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#			DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#			SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#			CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#			OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#			OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(AUXPORT_VERSION 0.1)
include(FetchContent)

# Fetching AuxPort
FetchContent_Declare(
	auxport
	GIT_REPOSITORY https://github.com/inpinseptipin/AuxPort
	GIT_TAG	master
)
FetchContent_MakeAvailable(auxport)

# Fetching uvgRTP
FetchContent_Declare(
	uvgrtp
	GIT_REPOSITORY https://github.com/ultravideo/uvgRTP
	GIT_TAG	master
)
FetchContent_MakeAvailable(uvgrtp)

# Fetching AudioFile
FetchContent_Declare(
	audiofile
	GIT_REPOSITORY https://github.com/adamstark/AudioFile.git
	GIT_TAG master
)
FetchContent_MakeAvailable(audiofile)



project(Client)
add_executable(
	${PROJECT_NAME}
	AuxRTP.cpp
	AuxRTP.h
	AuxAudioQueue.cpp
	AuxAudioQueue.h
	AuxWave.cpp
	AuxWave.h
	AuxSpectral.h
	main.cpp 
)



# Adding AuxPort include directories
target_include_directories(${PROJECT_NAME} PUBLIC "${AuxTest_SOURCE_DIR}/AuxPort")
# Linking AuxPort Libraries
target_link_libraries(${PROJECT_NAME} PUBLIC AuxPort)



# Adding LibTorch Header Files
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/vendor/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/vendor/include/torch/csrc/api/include")



# Adding LibTorch Static Libraries
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/asmjit.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/c10.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/clog.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/cpuinfo.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/dnnl.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/fbgemm.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/fbjni.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/fmtd.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/kineto.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/libprotobufd.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/libprotobuf-lited.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/libprotocd.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/pthreadpool.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/pytorch_jni.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/torch.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/torch_cpu.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/vendor/libs/Static/XNNPACK.lib)





# Adding uvgrtp include directories
target_include_directories(${PROJECT_NAME} PUBLIC "${uvgrtp_SOURCE_DIR}/include")
# Linking uvgrtp import Libraries and other required Librarires
target_link_libraries(${PROJECT_NAME} PUBLIC debug "${uvgrtp_BINARY_DIR}/Debug/uvgrtp.lib" optimized "${uvgrtp_BINARY_DIR}/Release/uvgrtp.lib")
target_link_libraries(${PROJECT_NAME} PUBLIC Ws2_32.lib)
target_link_libraries(${PROJECT_NAME} PUBLIC Advapi32.lib)


# Adding AudioFile include directories
target_include_directories(${PROJECT_NAME} PUBLIC "${audiofile_SOURCE_DIR}/")



	
