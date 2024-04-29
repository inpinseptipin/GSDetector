#include "AuxAudioQueue.h"

void AuxPort::AudioQueue::setBufferSize(uint32_t bufferSize)
{
	buffer.resize(bufferSize);
	waveFile.setBufferSize(bufferSize);
}

void AuxPort::AudioQueue::setFileNames(const std::vector<std::string>& fileNames)
{
	for (uint32_t i = 0; i < fileNames.size();i++)
		fileQueue.push(fileNames[i]);
}

std::vector<float>& AuxPort::AudioQueue::getBuffer()
{
	if (waveFile.isExhausted() && fileQueue.size() > 0)
	{
		waveFile.load(fileQueue.front());
		fileQueue.pop();
	}
	waveFile.getBuffer(buffer);
	return buffer;
		
}

bool AuxPort::AudioQueue::isExhausted()
{
	return fileQueue.size() == 0 && waveFile.isExhausted();
}

