#pragma once

#include <fstream>
#include <memory>

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

template<class TRootElement, class TIOStream>
class GenericOutputStore: boost::noncopyable
{
public:
	typedef std::vector<TRootElement *> WeakElementList;

	GenericOutputStore(TIOStream & iostream) :
			m_rawOut(&iostream), m_codedOut(&m_rawOut)
	{
	}

	void storeElement(TRootElement const& e)
	{
		const WeakElementList elist =
		{
		{ &e } };
		storeElements(elist);
	}

	void storeElements(WeakElementList const & el)
	{
		for (auto e : el)
		{
			std::string sBuffer;
			m_codedOut.WriteVarint32(uint32_t(EntryType::Event));

			e->SerializeToString(&sBuffer);

			m_codedOut.WriteVarint32(sBuffer.size());
			m_codedOut.WriteString(sBuffer);
		}
		// todo: flush the stream here ?

	}

private:
	::google::protobuf::io::OstreamOutputStream m_rawOut;
	::google::protobuf::io::CodedOutputStream m_codedOut;
};

template<class TRootElement, class TIOStream>
class GenericInputStore: boost::noncopyable
{
public:

	typedef std::vector<TRootElement *> WeakElementList;

	GenericInputStore(TIOStream & iostream) :
		m_rawIn(&iostream), m_codedIn(&m_rawIn)
	{
		// don't warn about loo long messages, this is HPC and not the web ...
		// we set a 10 GB limit here
		m_codedIn. SetTotalBytesLimit(INT_MAX,INT_MAX);

	}

	TRootElement * readNextElement()
	{
		WeakElementList elist;
		if (readNextElements(1, elist) != 1)
			throw std::runtime_error(
					"requested element could not be read from input");

		return elist[0];
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
			std::string sBuffer;

			// read message type
			uint32_t msgType;
			if (!m_codedIn.ReadVarint32(&msgType))
			{
				break;
			}
			EntryType etype = EntryType(msgType);
			if (etype != EntryType::Event)
			{
				throw std::runtime_error(
						"unsupported message type of number "
								+ boost::lexical_cast<std::string>(msgType));
			}

			uint32_t msgSize;
			if (!m_codedIn.ReadVarint32(&msgSize))
			{
				break;
			}

			if ((msgSize > 0) && (m_codedIn.ReadString(&sBuffer, msgSize)))
			{
				TRootElement * elem = new TRootElement();
				elem->ParseFromString(sBuffer);
				el.push_back(elem);
				elementsRead++;
			}
		}

		return elementsRead;
	}

private:
	::google::protobuf::io::IstreamInputStream m_rawIn;
	::google::protobuf::io::CodedInputStream m_codedIn;
};

typedef GenericInputStore< PEvent::PEvent, std::ifstream > EventStoreInput;
typedef GenericOutputStore< PEvent::PEvent, std::ofstream > EventStoreOutput;
