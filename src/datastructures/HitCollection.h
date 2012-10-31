#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

struct NullData
{
};

typedef std::pair<NullData, float> NullDataItem;

template<class TCollection>
class CollectionView
{
public:
	typedef typename TCollection::index_type index_type;

	CollectionView(TCollection & collection, index_type i) :
			m_col(collection), m_index(i)
	{

	}

	/*	typedef typename TCollection::key_type key_type;
	 typedef typename TCollection::data_type data_type;
	 */
	template<class TDataClass>
	typename TDataClass::data_type getValue() const
	{
		return m_col.getValue(TDataClass(), m_index);
	}

	template<class TDataClass>
	void setValue(typename TDataClass::data_type const& v)
	{
		m_col.setValue(TDataClass(), m_index, v);
	}

private:
	TCollection & m_col;
	index_type m_index;
};

// todo: can we have a Collection view, which looks like a vector type

template<typename THead = NullDataItem, typename ... TTail>
class dataitems: public dataitems<TTail ...>
{
public:

	typedef dataitems<THead, TTail ...> own_type;
	typedef CollectionView<own_type> iterator_type;
	typedef dataitems<TTail ...> inherited;

	typedef THead key_type;
	typedef typename THead::data_type data_type;

	template <typename T>
	using internal_collection = std::vector< T >;

	typedef int index_type;

	// necessary to unhide the set/get methods of the base classes
	using inherited::setValue;
	using inherited::getValue;

	dataitems()
	{
	}

	dataitems(index_type initialSize) :
			inherited(initialSize)
	{
		m_data.resize(initialSize);
	}

	void addItem ()
	{
		inherited::addItem();
		m_data.push_back( data_type () );
	}

	index_type size() const
	{
		return m_data.size();
	}

	//data_type getValue ( key_type ) const

	/*
	 template <key_type, data_type>
	 data_type getValue ( ) const
	 {
	 return m_data;
	 }
	 */
	data_type getValue(key_type, index_type i) const
	{
		return m_data[i];
	}

	// todo: no copy ...
	iterator_type getIterator()
	{
		return iterator_type(*this, index_type(0));
	}

	void setValue(key_type, index_type i, data_type const& v)
	{
		m_data[i] = v;
	}

private:
	key_type m_key;
	internal_collection<data_type> m_data;
};

// terminate the inheritance scheme
template<>
class dataitems<>
{
public:
	typedef int index_type;
	dataitems(index_type initialSize)
	{
	}

	void setValue();

	void addItem()0000

	/*template < class Key, class Data >
	 Data getValue() const;*/
	void getValue() const;
};

struct FloatItem
{
	typedef float data_type;
};

struct GlobalX: public FloatItem
{
};
struct GlobalY: public FloatItem
{
};

template <typename T>
using FloatDataItem = std::pair< T, float >;

typedef dataitems<GlobalX, GlobalY> HitDataItems;

class HitCollection: public HitDataItems
{
public:
	HitCollection(int items) :
			dataitems<GlobalX, GlobalY>(items)
	{

	}

	// bring the constructors of the dataitems in this scope
	// -> will first work in gcc 4.8
	//using HitDataItems::dataitems<GlobalX, GlobalY>;

	/*float getGlobalX () const
	 {
	 return getValue ( GlobalX() );
	 //return dataitems<  FloatDataItem<GlobalX> > ::m_data;
	 }*/

public:
	float dummy;
};

