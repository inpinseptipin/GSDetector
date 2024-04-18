#ifndef AUXPORT_RTP_H
#define AUXPORT_RTP_H

#include "uvgrtp/lib.hh"
#include "AuxPort.h"
#include "AuxWave.h"

namespace AuxPort
{
	namespace RTP
	{

		void rtp_receive_hook(void* arg, uvgrtp::frame::rtp_frame* frame);


		class Session
		{
		public:
			Session() = default;
			~Session();
			Session(const Session& session) = default;
			void attachContext(uvgrtp::context* context);
			void createSession(const std::string& remoteAddress);
			virtual void setFlags() = 0;
			std::string getRemoteAddress();
			void createStream(const uint16_t remotePort);
		protected:
			uvgrtp::context* context;
			uvgrtp::media_stream* mediaStream;
			std::string remoteAddress;
			uvgrtp::session* session;
			int flags = -1;
		};


		class Client : public Session,AuxPort::ILog
		{
		public:
			Client() = default;
			~Client() = default;
			Client(const Client& client) = default;
			void setFlags() override;
			bool run();
			void Log() override;
			std::vector<float> packetBuffer;
		private:
		};

		class Server : public Session
		{
		public:
			Server() = default;
			~Server() = default;
			Server(const Server& client) = default;
			void setFlags() override;
			void run();
		private:
		};


;

	}
}

#endif