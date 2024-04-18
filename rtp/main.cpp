#include <torch/torch.h>
#include <torch/script.h>
#include "AuxRTP.h"


int main(int argc,char* argv)
{
	/*AuxPort::WaveFile waveFile;
	waveFile.load("C:/Users/SatyarthArora/Downloads/audio.wav");
	waveFile.Log();*/

	torch::jit::script::Module module;
	try
	{
		module = torch::jit::load("C:/ROOT/AuxPort/AuxPort_rtp/master/Models/traced_resnet18.pt");
	}
	catch(const c10::Error& e)
	{
		AuxPort::Logger::Log("Cannot Load Model", AuxPort::LogType::Error);
	}
	AuxPort::Logger::Log("Model Loaded Successfully", AuxPort::LogType::Success);


	std::vector<torch::jit::IValue> inputs;
	auto data = AuxPort::Utility::generateRandomValues<float>(256);

	module.eval();
	
	auto column = torch::tensor({ data });

	auto row = column.transpose(0,1);

	auto tensor = row.dot(column);
	

	
	inputs.push_back(tensor);

	auto output = module.forward(inputs).toTensor();

	output = output.sigmoid_();

	AuxPort::Logger::Log(output);

}