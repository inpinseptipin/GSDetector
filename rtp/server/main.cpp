#include "AuxRTP.h"
#include "AuxSpectral.h"
#include "AuxAudioQueue.h"

int main(int argc, char* argv)
{
	uvgrtp::context context;
	AuxPort::RTP::Server server;
	server.attachContext(&context);
	server.createSession("127.0.0.1");
	server.setFlags();
	server.createStream(3333);
	server.setFolders({ "C:\\ROOT\\AuxPort\\Dataset" });
	
	server.run();
}