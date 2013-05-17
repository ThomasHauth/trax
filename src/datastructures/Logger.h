#pragma once

#include <boost/noncopyable.hpp>
#include <vector>
#include <iostream>
#include <sstream>

class Logger : private boost::noncopyable {

#define LOG  Logger::getInstance().addLogEntry(Logger::cNORMAL)
#define VLOG Logger::getInstance().addLogEntry(Logger::cVERBOSE)
#define PLOG Logger::getInstance().addLogEntry(Logger::cPROLIX)

#define VERBOSE Logger::getInstance().getLogLevel() >= Logger::cVERBOSE
#define PROLIX  Logger::getInstance().getLogLevel() >= Logger::cPROLIX

public:

	typedef std::pair<int, std::stringstream*> tLogEntry;

	static Logger& getInstance();

	int getLogLevel() const { return logLevel; }
	void setLogLevel(int l) { logLevel = l; }

	std::ostream & addLogEntry(uint level);

	std::ostream& printLog(std::ostream & out);

	std::ostream& printLog(std::ostream & out, int level);

	~Logger();


private:
	Logger() : logLevel(Logger::cNORMAL) {};

	std::vector<tLogEntry> logEntries;
	int logLevel;

public:
	static constexpr int cSILENT = -1;
	static constexpr int cNORMAL = 0;
	static constexpr int cVERBOSE = 1;
	static constexpr int cPROLIX = 2;

};

std::ostream & operator<<(std::ostream& s, const Logger & log);
