#ifndef AUXAUDIOQUEUE_H
#define AUXAUDIOQUEUE_H
#include "AuxPort.h"
#include "AuxWave.h"
#include <queue>

namespace AuxPort
{
	class AudioQueue
	{
	public:
		AudioQueue() = default;
		~AudioQueue() = default;
		AudioQueue(const AudioQueue& audioQueue) = default;
		void setBufferSize(uint32_t bufferSize);
		void setFileNames(const std::vector<std::string>& fileNames);
		std::vector<float>& getBuffer();
		bool isExhausted();
	private:
		WaveFile waveFile;
		std::vector<float> buffer;
		std::queue<std::string> fileQueue;
	};
}

#endif