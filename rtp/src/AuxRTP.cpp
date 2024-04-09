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
	AuxPort::Logger::Log("Received RTP Frame, Payload Size : " + frame->payload_len);

	

	(void)uvgrtp::frame::dealloc_frame(frame);
}


void AuxPort::RTP::Client::setFlags()
{
	flags = RCE_FRAGMENT_GENERIC | RCE_RECEIVE_ONLY;
}



bool AuxPort::RTP::Client::run()
{
	AuxPort::Logger::Log("Client ready to Retrieve Data");
	if (!session || mediaStream->install_receive_hook(nullptr, rtp_receive_hook) != RTP_OK)
	{
		std::cerr << "Failed to install RTP receive hook!" << std::endl;
		return 0;
	}

}

void AuxPort::RTP::Server::setFlags()
{
	flags = RCE_FRAGMENT_GENERIC | RCE_SEND_ONLY;
}

void AuxPort::RTP::Server::run()
{
	auto media = std::unique_ptr<uint8_t[]>(new uint8_t[512]);
	uint32_t count = 0;
	AuxPort::Logger::Log("Server ready to Send Data");
	
	while (true)
	{
		if (send)
		{
			auto vector = AuxPort::Utility::generateRandomValues<float>(128);
			auto vecAsUint = (uint8_t*)vector.data();
			for (uint32_t i = 0; i < vector.size() * 4; i++)
				media[i] = vecAsUint[i];
			AuxPort::Logger::Log("Packet Sent : " + AuxPort::Casters::toStdString(count));
			if (mediaStream->push_frame(media.get(), 512, RTP_NO_FLAGS) != RTP_OK)
			{
				std::cerr << "Failed to send frame!" << std::endl;
			}
		}
	}
		
		
	
}
