#include <torch/torch.h>
#include <torch/script.h>
#include "AuxRTP.h"
#include "AuxSpectral.h"
#include "AuxAudioQueue.h"

int main(int argc, char* argv)
{

	torch::jit::script::Module module;
	try
	{
		module = torch::jit::load("C:/ROOT/AuxPort/AuxPort_rtp/master/Models/traced_resnet18.pt");
	}
	catch (const c10::Error& e)
	{
		AuxPort::Logger::Log("Cannot Load Model", AuxPort::LogType::Error);
	}
	AuxPort::Logger::Log("Model Loaded Successfully", AuxPort::LogType::Success);

	

	AuxPort::Audio::Spectral::Spectrogram<float> spectrogram(512, AuxPort::Audio::Window::HannWin, 256, 256);
	AuxPort::AudioQueue audioQueue;
	audioQueue.setBufferSize(256);
	audioQueue.setFileNames({"audio.wav"});
	auto tensor = torch::zeros({ 1,1,256,256 });
	module.eval();
	std::vector<float> zeroBuffer;
	zeroBuffer.resize(256);
	std::fill(zeroBuffer.begin(), zeroBuffer.end(), 0.0f);

	std::vector<torch::jit::IValue> inputs;

	std::vector<float> buffer;
	while (true)
	{
		for (uint32_t i = 0; i < 256; i++)
		{
			buffer = audioQueue.isExhausted() ? zeroBuffer : audioQueue.getBuffer();
			auto specData = spectrogram.processBuffer(buffer);
			for (uint32_t j = 0; j < 256; j++)
			{
				tensor[0][0][i][j] = AuxPort::Utility::linearTodB(abs(specData[j].real()));
			}
		}
		inputs.push_back(tensor);
		auto output = module.forward(inputs).toTensor();
		inputs.pop_back();
		output = output.sigmoid_();
		AuxPort::Logger::Log(output);
	}

	
	


	







}