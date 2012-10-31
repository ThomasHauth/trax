#pragma once

//#include <tuple>
#include <utility>
#include <boost/type_traits.hpp>

struct NullData
{
};

typedef std::pair < NullData, float > NullDataItem;


template<typename THead = NullDataItem, typename ... TTail>
class dataitems: public dataitems<TTail ...> {
public:

	typedef dataitems<TTail ...> inherited;

	typedef typename THead::first_type key_type;
	typedef typename THead::second_type data_type;

	// necessary to unhide the set/get methods of the base classes
	using inherited::setValue;
	using inherited::getValue;

	//data_type getValue ( key_type ) const

/*
	template <key_type, data_type>
	data_type getValue ( ) const
	{
		return m_data;
	}
*/
	data_type getValue (key_type ) const
	{
		return m_data;
	}

	void setValue ( key_type, data_type const& v )
	{
		m_data = v;
	}

private:
	typename THead::first_type m_key;
	typename THead::second_type m_data;
};

// terminate the inheritance scheme
template<>
class dataitems<> {
public:
	void setValue();

	/*template < class Key, class Data >
	Data getValue() const;*/
	void getValue() const;
};

struct GlobalX {
};
struct GlobalY {
};

template <typename T>
using FloatDataItem = std::pair< T, float >;

class HitCollection: public dataitems<FloatDataItem<GlobalY>,
		FloatDataItem<GlobalX> > {

	/*float getGlobalX () const
	{
		return getValue ( GlobalX() );
		//return dataitems<  FloatDataItem<GlobalX> > ::m_data;
	}*/

public:
	float dummy;
};

