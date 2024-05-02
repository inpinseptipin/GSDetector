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
		AudioQueue();
		~AudioQueue() = default;
		AudioQueue(const AudioQueue& audioQueue) = default;
		void addBufferToQueue(float* data, uint32_t size);
		std::vector<float> getBuffer();
		bool isExhausted();
	private:
		std::queue<std::vector<float>> audioBuffer;
		std::vector<float> twoSecondBuffer;
		uint32_t bufferCounter = 0;

	};
}

#endif