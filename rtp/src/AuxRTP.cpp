#include "AuxRTP.h"

void AuxPort::Session::createSession(const std::string& remoteAddress, uint16_t remortPort)
{
	this->remoteAddress = remoteAddress;
	this->remotePort = remortPort;
	
}

void AuxPort::Session::setFlag(const Flags& flags)
{

}

