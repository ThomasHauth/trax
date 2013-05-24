#pragma once

#include <fstream>
#include <memory>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

#include <climits>

#include <boost/noncopyable.hpp>
#include <boost/lexical_cast.hpp>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
//#include <boost/ptr_container/>

#include "Event.pb.h"

enum class EntryType
{
	Event = 1
};

template<class TRootElement>
class GenericOutputStore: boost::noncopyable
{
public:
	typedef std::vector<const TRootElement *> WeakElementList;

	GenericOutputStore(const std::string & filename, bool append = false)
	{
		file = filename;

		if(append && fileExists(file)){

			//read number of present events
			fd = open(filename.c_str(), O_RDONLY);

			google::protobuf::io::FileInputStream rawIn(fd);
			google::protobuf::io::CodedInputStream codedIn(&rawIn);

			codedIn.SetTotalBytesLimit(INT_MAX,INT_MAX);
			codedIn.ReadLittleEndian32(&eventsWritten);

			rawIn.Close();
			close(fd);

			fd = open(filename.c_str(), O_WRONLY | O_APPEND);

			m_rawOut = new google::protobuf::io::FileOutputStream(fd);
			m_codedOut = new google::protobuf::io::CodedOutputStream(m_rawOut);

		} else {
			fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC);

			m_rawOut = new google::protobuf::io::FileOutputStream(fd);
			m_codedOut = new google::protobuf::io::CodedOutputStream(m_rawOut);
			eventsWritten = 0;
			m_codedOut->WriteLittleEndian32(0); //to be later overwritten
		}
	}

	~GenericOutputStore(){
		Close();
	}

	void storeElement(TRootElement const& e)
	{
		std::string sBuffer;
		m_codedOut->WriteVarint32(uint32_t(EntryType::Event));

		e.SerializeToString(&sBuffer);

		m_codedOut->WriteVarint32(sBuffer.size());
		m_codedOut->WriteString(sBuffer);

		++eventsWritten;
	}

	void storeElements(WeakElementList const & el)
	{
		for (auto e : el)
		{
			storeElement(*e);
		}
	}

	uint getEvents(){
		return eventsWritten;
	}

	void Close(){

		if(eventsWritten > 0){ //more than zero events are written -> update written events;
			//close file
			delete m_codedOut;
			m_rawOut->Close();
			delete m_rawOut;
			close(fd);
			//reopen
			fd = open(file.c_str(), O_WRONLY);
			m_rawOut = new google::protobuf::io::FileOutputStream(fd);
			m_codedOut = new google::protobuf::io::CodedOutputStream(m_rawOut);
			m_codedOut->WriteLittleEndian32(eventsWritten);

			eventsWritten = 0;
		}

		if(m_codedOut != NULL){

			delete m_codedOut; m_codedOut = NULL;
		}

		if(m_rawOut != NULL){
			m_rawOut->Close();
			delete m_rawOut; m_rawOut = NULL;
		}
		close(fd);
	}



private:

	std::string file;
	int fd;
	::google::protobuf::io::FileOutputStream * m_rawOut;
	::google::protobuf::io::CodedOutputStream * m_codedOut;
	uint eventsWritten;

	inline bool fileExists (const std::string& name) {
		struct stat buffer;
		return (stat (name.c_str(), &buffer) == 0);
	}

};

template<class TRootElement>
class GenericInputStore: boost::noncopyable
{
public:

	typedef std::vector<TRootElement *> WeakElementList;

	GenericInputStore(const std::string & filename)
	{
		fd = open(filename.c_str(), O_RDONLY);
		m_rawIn = new google::protobuf::io::FileInputStream(fd);
		m_codedIn = new google::protobuf::io::CodedInputStream(m_rawIn);
		m_codedIn->SetTotalBytesLimit(INT_MAX,INT_MAX);
		m_codedIn->ReadLittleEndian32(&nEvents);
		eventsRead = 0;
	}

	~GenericInputStore(){
		Close();
	}

	TRootElement * readNextElement()
	{

		TRootElement * elem = NULL;

		if(eventsRead < nEvents){

			std::string sBuffer;

			// read message type
			uint32_t msgType;
			if (!m_codedIn->ReadVarint32(&msgType))
			{
				return elem;
			}
			EntryType etype = EntryType(msgType);
			if (etype != EntryType::Event)
			{
				return elem;
			}

			uint32_t msgSize;
			if (!m_codedIn->ReadVarint32(&msgSize))
			{
				return elem;
			}

			if ((msgSize > 0) && (m_codedIn->ReadString(&sBuffer, msgSize)))
			{
				elem = new TRootElement();
				elem->ParseFromString(sBuffer);
				eventsRead++;
			}
		}

		return elem;
	}

	/*
	 * Reads the next n elements and puts them in the el list. Returned is the
	 * number of actual elements read
	 */
	size_t readNextElements(size_t n, WeakElementList & el)
	{
		unsigned int elementsRead = 0;

		// start reading from the stream the messages we need
		for (uint32_t i = 0; i < n; ++i)
		{

			TRootElement * element = readNextElement();
			if(element != NULL){
				el.push_back(element);
				elementsRead++;
			}
		}

		return elementsRead;

	}

	uint getEvents(){
		return nEvents;
	}

	void Close(){
		if(m_codedIn != NULL){
			delete m_codedIn; m_codedIn = NULL;
		}
		if(m_rawIn != NULL){
			m_rawIn->Close();
			delete m_rawIn; m_rawIn = NULL;
		}
		close(fd);
	}

private:
	int fd;
	::google::protobuf::io::FileInputStream * m_rawIn;
	::google::protobuf::io::CodedInputStream * m_codedIn;
	uint nEvents;
	uint eventsRead;
};

typedef GenericInputStore<PB_Event::PEvent> EventStoreInput;
typedef GenericOutputStore<PB_Event::PEvent> EventStoreOutput;
