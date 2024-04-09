#include "AuxRTP.h"


int main()
{
	AuxPort::RTP::Server server;
	AuxPort::RTP::Client client;

	uvgrtp::context context;

	server.attachContext(&context);
	client.attachContext(&context);
	server.setFlags();
	client.setFlags();

	server.createSession("127.0.0.1");
	client.createSession("127.0.0.1");

	

	server.createStream(3333);
	client.createStream(3333);

	server.run();
	client.run();

}