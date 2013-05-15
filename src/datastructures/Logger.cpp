/*
 * Logger.cpp
 *
 *  Created on: May 15, 2013
 *      Author: dfunke
 */

#include "Logger.h"

Logger& Logger::getInstance()
{
	static Logger    instance; // Guaranteed to be destroyed.
	// Instantiated on first use.
	return instance;
}

std::stringstream & Logger::addLogEntry(uint level) {
	std::stringstream* s = new std::stringstream();
	tLogEntry entry = std::make_pair(level, s);
	logEntries.push_back(entry);

	return *(logEntries[logEntries.size()-1].second);
}

std::ostream & Logger::printLog(std::ostream & out){
	return printLog(out, logLevel);
}

std::ostream & Logger::printLog(std::ostream & out, int level){

	//shortcut SILENT processing
	if(level == Logger::cSILENT)
		return out;

	for(tLogEntry t : logEntries){
		if( t.first <= level){
			out << t.second->str();
		}
	}

	return out;
}

Logger::~Logger(){
	for(tLogEntry t : logEntries){
		delete t.second;
	}
}

std::ostream & operator<<(std::ostream& s, const Logger & log){

	return Logger::getInstance().printLog(s);
}



