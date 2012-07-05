#include <cstdio>
#include <cstdlib>
#ifndef _WIN32
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

#include <iostream>
#include <sstream>
#include <string>

#include "ipcsocket.h"

using namespace std;

ipcsocket::ipcsocket()
: conn(false)
{
#ifdef _WIN32
	pipename = "\\\\.\\pipe\\svspipe";
#endif
}

ipcsocket::~ipcsocket() {
	if (conn) {
		disconnect();
	}
}

bool ipcsocket::connect(const string &path) {
#ifndef _WIN32
	socklen_t len;
	struct sockaddr_un addr;
	
	bzero((char *) &addr, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strcpy(addr.sun_path, path.c_str());
	len = strlen(addr.sun_path) + sizeof(addr.sun_family) + 1;
	
	if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
		perror("ipcsocket::ipcsocket");
		exit(1);
	}
	
	if (::connect(fd, (struct sockaddr *) &addr, len) == -1) {
		return false;
	}
#else
	pipe = CreateFile(pipename, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
	
	if (pipe == INVALID_HANDLE_VALUE)
	{
		std::cout << "ipcsocket::ipcsocket (Creating Pipe): " << GetLastError() << std::endl;
		return false;
	}
	
	DWORD mode = PIPE_READMODE_MESSAGE;
	
	bool success = SetNamedPipeHandleState(pipe, &mode, NULL, NULL);
	
	if (!success)
	{
		std::cout << "ipcsocket::ipcsocket (Setting Message Read Mode on Pipe): " << GetLastError() << std::endl;
		return false;
	}
#endif
	
	conn = true;
	return true;
}

void ipcsocket::disconnect() {
#ifndef _WIN32
	close(fd);
#else
	CloseHandle(pipe);
#endif
	conn = false;
}

bool ipcsocket::send(const string &s) {
#ifndef _WIN32
	int n;
#else
	DWORD n;
#endif
	
	if (!conn) return false;
	
	string t = s;
	
	while (t.size() > 0)
	{
#ifndef _WIN32
		if ((n = ::send(fd, t.c_str(), t.size(), 0)) <= 0)
		{
			if (errno != EINTR)
			{
				disconnect();
				return false;
			}
		}
#else
		bool success = WriteFile(pipe, t.c_str(), t.size(), &n, NULL);
		
		if (!success)
		{
			disconnect();
			return false;
		}
#endif
		else
			t.erase(0, n);
	}
}
