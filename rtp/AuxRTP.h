#ifndef AUXPORT_RTP_H
#define AUXPORT_RTP_H

#include "uvgrtp/lib.hh"

namespace AuxPort
{
	class Session
	{
	public:
		enum Flags
		{
			sendOnly,receiveOnly,sendAndReceive
		};
		Session() = default;
		~Session() = default;
		void createSession(const std::string& remoteAddress, uint16_t remortPort);
		void setFlag(const Flags& flags = Flags::sendOnly);
		Session(const Session& session) = default;
	private:
		std::string remoteAddress;
		uint16_t remotePort;
		Flags flag;
	};
}

#endif