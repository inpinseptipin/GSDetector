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
		void load(const std::string& fileNameWithPath);
		void Log() override;
	private:
		AudioFile<float> file;


	};
}

#endif