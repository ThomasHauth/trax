#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"

struct Radius: public clever::FloatItem
{
};

#define DICTIONARY_ITEMS Radius

typedef clever::Collection<DICTIONARY_ITEMS> DictionaryItems;

class Dictionary: public DictionaryItems
{
public:
	typedef DictionaryItems dataitems_type;

	Dictionary()
	{

	}

	Dictionary(int items) :
			clever::Collection<DICTIONARY_ITEMS>(items)
	{

	}

public:
	clever::OpenCLTransfer<DICTIONARY_ITEMS> transfer;
};

class DictionaryEntry: private clever::CollectionView<Dictionary>
{
public:
// get a pointer to one hit in the collection
	DictionaryEntry(Dictionary & collection, index_type i) :
			clever::CollectionView<Dictionary>(collection, i)
	{

	}

// create a new hit in the collection
	DictionaryEntry(Dictionary & collection) :
			clever::CollectionView<Dictionary>(collection)
	{
	}

	float radius() const
	{
		return getValue<Radius>();
	}

};

