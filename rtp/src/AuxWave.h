#ifndef AUXWAVE_H
#define AUXWAVE_H

#include "AudioFile.h"
#include "AuxPort.h"

namespace AuxPort
{
	class WaveFile : public AuxPort::ILog
	{
	public:
		WaveFile() = default;
		~WaveFile() = default;
		WaveFile(const WaveFile& waveFile) = default;
		bool load(const std::string& fileNameWithPath);
		void getBuffer(std::vector<float>& buffer);
		bool isExhausted();
		void Log() override;
		void setBufferSize(uint32_t bufferSize);
	private:
		float getSample();
		AudioFile<float> file;
		size_t bufferSize = 256;
		size_t numberOfBuffers=0;
		uint32_t zeroPadAmount=0;
		size_t currentBuffer=0;
		size_t counter=0;
	};

	
}

#endif