#ifndef IPCSOCKET_H
#define IPCSOCKET_H

#include <string>

#ifdef _WIN32
#include "portability.h"
#endif

class ipcsocket {
public:
	ipcsocket();
	~ipcsocket();
	
	bool send(const std::string &s);
	
	bool connect(const std::string &path);
	
	bool connected() const { return conn; }
	void disconnect();
	
private:
#ifndef _WIN32
	int fd;
#else
	HANDLE pipe;
	const char* pipename;
#endif
	
	bool conn;
};

#endif
