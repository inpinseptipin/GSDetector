
#include "AuxRTP.h"
#include "AuxSpectral.h"
#include "AuxAudioQueue.h"

int main(int argc, char* argv)
{

	//torch::jit::script::Module module;
	//try
	//{
	//	module = torch::jit::load("C:\\ROOT\\AuxPort\\GSDetect\\master\\detection\\trained_models\\compiled\\EnhancedCNN.ptc");
	//}
	//catch (const c10::Error& e)
	//{
	//	AuxPort::Logger::Log("Cannot Load Model", AuxPort::LogType::Error);
	//	return 0;
	//}
	//AuxPort::Logger::Log("Model Loaded Successfully", AuxPort::LogType::Success);

	//

	//AuxPort::Audio::Spectral::Spectrogram<float> spectrogram(256, AuxPort::Audio::Window::HannWin, 128, 128);
	//AuxPort::AudioQueue audioQueue;
	//AuxPort::Directory directory;
	//directory.setDirectory("C:\\Users\\SatyarthArora\\Downloads\\7004819\\edge-collected-gunshot-audio\\edge-collected-gunshot-audio\\glock_17_9mm_caliber");
	//auto listOfFiles = directory.getListOfFiles(".wav");
	//audioQueue.setBufferSize(128);
	//audioQueue.setFileNames(listOfFiles);
	//auto tensor = torch::zeros({ 1,1,256,256 });
	//std::vector<float> zeroBuffer;
	//zeroBuffer.resize(256);
	//std::fill(zeroBuffer.begin(), zeroBuffer.end(), 0.0f);
	//
	//std::vector<torch::jit::IValue> inputs;

	//std::vector<float> buffer;
	//

	//module.eval();

	//while (true)
	//{
	//	for (uint32_t i = 0; i < 256; i++)
	//	{
	//		buffer = audioQueue.isExhausted() ? zeroBuffer : audioQueue.getBuffer();
	//		auto specData = spectrogram.processBuffer(buffer);
	//		for (uint32_t j = 0; j < 256; j++)
	//		{
	//			tensor[0][0][i][j] = AuxPort::Utility::linearTodB(abs(pow(specData[j],2)));
	//		}
	//	}
	//	
	//	inputs.push_back(tensor);
	//	auto output = module.forward(inputs).toTensor();
	//	inputs.pop_back();
	//	output = output.sigmoid_();
	//	AuxPort::Logger::Log(output);
	//}

	
	


	AuxPort::RTP::Client client;
	uvgrtp::context context;
	client.attachContext(&context);
	client.createSession("127.0.0.1");
	client.setFlags();
	client.createStream(3333);
	client.run();








}