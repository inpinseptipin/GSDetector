#include "AuxAudioQueue.h"


AuxPort::AudioQueue::AudioQueue()
{
	bufferCounter = 0;
	twoSecondBuffer.resize(88200);
}

void AuxPort::AudioQueue::addBufferToQueue(float* data, uint32_t size)
{
	for (uint32_t i = 0; i < size; i++)
	{
		if (bufferCounter <= twoSecondBuffer.size() - 1)
			twoSecondBuffer[bufferCounter++] = data[i];
		else
		{
			audioBuffer.push(twoSecondBuffer);
			bufferCounter = 0;
		}
	}
}

std::vector<float> AuxPort::AudioQueue::getBuffer()
{
	auto buffer = audioBuffer.front();
	audioBuffer.pop();
	return buffer;
}

bool AuxPort::AudioQueue::isExhausted()
{
	return audioBuffer.empty();
}
