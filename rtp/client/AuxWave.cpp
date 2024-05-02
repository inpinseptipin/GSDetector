#include "AuxWave.h"
#include <exception>

bool AuxPort::WaveFile::load(const std::string& fileNameWithPath)
{
	AuxAssert(fileNameWithPath.size() > 0, "Invalid Path");
	try
	{
		AuxAssert(file.load(fileNameWithPath), "Invalid File Name");
		AuxAssert(file.samples.size() > 0, "Invalid Wave File");
		float tempNumberOfBuffers = static_cast<float>(file.samples[0].size()) / bufferSize;
		auto zeroPad = tempNumberOfBuffers - static_cast<uint32_t>(tempNumberOfBuffers);
		zeroPadAmount = (1-zeroPad) * bufferSize;
		numberOfBuffers = zeroPadAmount > 0 ? static_cast<size_t>(tempNumberOfBuffers)+1 : static_cast<size_t>(tempNumberOfBuffers);
		counter = 0;
		currentBuffer = 0;
	}
	catch (std::exception exception)
	{
		AuxPort::Logger::Log("Cannot Load File : " + fileNameWithPath);
	}
	
}

void AuxPort::WaveFile::getBuffer(std::vector<float>& buffer)
{
	if (!isExhausted())
	{
		if (zeroPadAmount && currentBuffer == numberOfBuffers - 1)
		{
			for (uint32_t i = 0; i < buffer.size(); i++)
				buffer[i] = counter >= file.samples[0].size() - 1 ? 0.0f : getSample();
			counter = 0;
		}
		else
		{
			for (uint32_t i = 0; i < buffer.size(); i++)
				buffer[i] = getSample();
		}
		currentBuffer++;
		
	}
}

bool AuxPort::WaveFile::isExhausted()
{
	return currentBuffer == numberOfBuffers;
}



void AuxPort::WaveFile::Log()
{
	setColour(AuxPort::ColourType::Blue);
	file.printSummary();
	std::cout << "Number Of Buffers : " << numberOfBuffers << "\n";
	std::cout << "Buffer Size : " << bufferSize << "\n";
	std::cout << "Zero Pad Amount : " << zeroPadAmount << "\n";
	std::cout << "|======================================|\n";
	setColour(AuxPort::ColourType::White);
}

void AuxPort::WaveFile::setBufferSize(uint32_t bufferSize)
{
	this->bufferSize = bufferSize;
}

float AuxPort::WaveFile::getSample()
{
	auto sample =  file.samples.size() > 1 ? (0.5f * file.samples[0][counter] + 0.5f * file.samples[1][counter]) : file.samples[0][counter];
	counter++;
	return sample;
}
