#ifndef AUXPORT_RTP_H
#define AUXPORT_RTP_H

#include "uvgrtp/lib.hh"
#include "AuxPort.h"
#include "AuxAudioQueue.h"
#include "Utility.h"

namespace AuxPort
{
	namespace RTP
	{




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




		class Server : public Session
		{
		public:
			Server() = default;
			~Server() = default;
			Server(const Server& client) = default;
			void setFlags() override;
			void run();
			void setFolders(const std::vector<std::string> folderNames);
		private:
			AudioQueue audioQueue;
			AuxPort::Directory directory;
		};
	}



}

#endif