#ifndef LOGGER_H
#define LOGGER_H

#include <ostream>
#include <sstream>
#include "cliproxy.h"

/*
 Don't forget to update logger_info in logger.cpp when you add new log types.
*/
enum log_type {
	LOG_ERR,
	LOG_CTRL,
	LOG_EM,
	LOG_SGEL,
	LOG_FOIL,
	NUM_LOG_TYPES,
};


class logger : public bool_proxy {
public:
	logger();
	
	template<class T>
	logger &operator<<(const T& v) {
		if (active) {
			ss << v;
		}
		return *this;
	}
	
	void init(const std::string &prefix, bool active);
	logger& operator<<(std::ostream& (*f)(std::ostream&));
	void proxy_use_sub(const std::vector<std::string> &args, std::ostream &os);
	
private:
	std::stringstream ss;
	std::string prefix;
	bool active;
};

class logger_set : public cliproxy {
public:
	logger_set();
	
	logger &get(log_type t) { return loggers[t]; }
	void proxy_get_children(std::map<std::string, cliproxy*> &c);
	
private:
	logger loggers[NUM_LOG_TYPES];
};

#endif
