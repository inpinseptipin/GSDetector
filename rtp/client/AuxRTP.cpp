#include "AuxRTP.h"
#include "AuxPort.h"
AuxPort::RTP::Session::~Session()
{
	session->destroy_stream(mediaStream);
	context->destroy_session(session);	
}

void AuxPort::RTP::Session::attachContext(uvgrtp::context* context)
{
	this->context = context;
}

void AuxPort::RTP::Session::createSession(const std::string& remoteAddress)
{
	this->remoteAddress = remoteAddress;
	session = context->create_session(remoteAddress);
}

std::string AuxPort::RTP::Session::getRemoteAddress()
{
	return remoteAddress;
}

void AuxPort::RTP::Session::createStream(const uint16_t remotePort)
{
	AuxAssert(flags != -1, "Flags have not set");
	mediaStream = session->create_stream(remotePort, RTP_FORMAT_GENERIC, flags);
}

void AuxPort::RTP::rtp_receive_hook(void* arg, uvgrtp::frame::rtp_frame* frame)
{
	Client* client = static_cast<Client*>(arg);

	float* data = reinterpret_cast<float*>(frame->payload);

	client->audioQueue.addBufferToQueue(data, frame->payload_len / 4);



	(void)uvgrtp::frame::dealloc_frame(frame);
}


void AuxPort::RTP::Client::setFlags()
{
	flags = RCE_FRAGMENT_GENERIC | RCE_RECEIVE_ONLY;
}



bool AuxPort::RTP::Client::run()
{
	AuxPort::Logger::Log("Client ready to Retrieve Data");
	if (!session || mediaStream->install_receive_hook(this, rtp_receive_hook) != RTP_OK)
	{
		std::cerr << "Failed to install RTP receive hook!" << std::endl;
		return 0;
	}
	while (true)
	{
		Log();
	}

}

void AuxPort::RTP::Client::Log()
{
	if (!audioQueue.isExhausted())
	{
		auto buffer = audioQueue.getBuffer();
		AuxPort::Logger::Log(bufferID++);
	}
}


