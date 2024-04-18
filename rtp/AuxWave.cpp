#include "AuxWave.h"

void AuxPort::WaveFile::load(const std::string& fileNameWithPath)
{
	AuxAssert(fileNameWithPath.size() > 0, "Invalid Path");
	AuxAssert(file.load(fileNameWithPath),"Invalid File Name");
}

void AuxPort::WaveFile::Log()
{
	AuxAssert(file.getLengthInSeconds() > 0, "No audio file is loaded");
	setColour(AuxPort::ColourType::Blue);
	file.printSummary();
	setColour(AuxPort::ColourType::White);

}
