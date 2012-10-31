#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

template<class TType>
class TypeToCLType
{
public:
	static std::string str();
};

template<>
class TypeToCLType<float>
{
public:
	typedef cl_float cl_type;
	static std::string str()
	{
		return "float";
	}
};

/*
 template<>
 std::string TypeToCLType<float>::str() {
 return "float";
 }

 template<>
 std::string TypeToCLType<CustomVector3>::str() {
 return "float3";
 }*/

struct NullData
{
};

typedef std::pair<NullData, float> NullDataItem;

template<class TCollection>
class CollectionView
{
public:
	typedef typename TCollection::index_type index_type;

	// point to somewhere
	CollectionView(TCollection & collection, index_type i) :
			m_col(collection), m_index(i)
	{

	}

	// add entry
	CollectionView(TCollection & collection) :
			m_col(collection)
	{
		m_index = m_col.addEntry();
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

	// iterator implementation, not done yet
	// pre op
	CollectionView& operator++()
	{
		m_index++;
		return *this;
	}

	// post op
	CollectionView operator++(int)
	{
		CollectionView tmp(*this); //Kopier-Konstruktor
		++(*this); //Inkrement
		return tmp; //alten Wert zurueckgeben
	}

protected:
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

	template<class TDataHead, class ... TDataTail>
	index_type addWithValue(TDataHead thisValue, TDataTail ... tail)
	{
		inherited::addWithValue(tail...);
		m_data.push_back(thisValue);
		return (m_data.size() - 1);
	}

	//template < class TItemHead
	index_type addEntry()
	{
		inherited::addEntry();
		m_data.push_back(data_type());
		return (m_data.size() - 1);
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

	internal_collection<data_type> & getRawBuffer()
	{
		return m_data;
	}

	internal_collection<data_type> const& getRawBuffer() const
	{
		return m_data;
	}

private:
	key_type m_key;
	internal_collection<data_type> m_data;
};

// terminate the inheritance scheme
template<>
class dataitems<> : private boost::noncopyable
{
public:
	typedef int index_type;
	dataitems()
	{
	}

	dataitems(index_type initialSize)
	{
	}

	void setValue();

	index_type addEntry()
	{
		return 0;
	}

	index_type addWithValue()
	{
		return 0;
	}

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
	typedef HitDataItems dataitems_type;

	HitCollection()
	{

	}

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

};

class GlobalPosition: private CollectionView<HitCollection>
{
public:
	// get a pointer to one hit in the collection
	GlobalPosition(HitCollection & collection, index_type i) :
			CollectionView<HitCollection>(collection, i)
	{

	}

	float x() const
	{
		return getValue<GlobalX>();
	}

	float y() const
	{
		return getValue<GlobalY>();
	}

	void setX(float v)
	{
		setValue<GlobalX>(v);
	}

	void setY(float v)
	{
		setValue<GlobalY>(v);
	}
	/*
	 * a bit difficult, where do we store to ?
	 GlobalPosition & GlobalPosition::operator+=(const GlobalPosition &rhs) {
	 rs
	 }*/
};

template<typename THead = NullDataItem, typename ... TTail>
class OpenCLTransfer: public OpenCLTransfer<TTail ...>	//, private boost::noncopyable
{
public:
	typedef dataitems<THead, TTail ...> dataitems_type;
	typedef dataitems<TTail ...> dataitems_tail;

	typedef OpenCLTransfer<TTail ...> inherited;

	void initBuffers(clever::context & context,
			dataitems_type const& collection)
	{
		// a bit uncool, every data row has a pointer to the context ...
		typedef TypeToCLType<typename THead::data_type> Conversion;
		typedef typename Conversion::cl_type cl_type;
		// init the buffer for the head

		const size_t bufferSize = sizeof(cl_type) * collection.size();
		m_buffer = context.create_buffer(bufferSize);

		std::cout << "Initializing buffer of type " << Conversion::str()
				<< " of size " << bufferSize << " bytes" << std::endl;

		// handle the tail of the collection
		inherited::initBuffers(context,
				static_cast<dataitems_tail const&>(collection));
	}

	/*
	 * 	std::vector <float> float_in ( elements, val );
	 std::vector <float> float_out ( elements, 0.0f );

	 cl_mem buffer = context.create_buffer (  sizeof(cl_float) * elements );

	 context.transfer_from_buffer  ( buffer, &float_in.front(),  sizeof(cl_float) * elements );
	 context.transfer_to_buffer  ( buffer, &float_out.front(),  sizeof(cl_float) * elements );
	 context.release_buffer(buffer);
	 *
	 */

// todo: can we do const here ?
	/*	void toDevice(dataitems_type & collection)
	 {
	 m_context.transfer_to_buffer(m_buffer,
	 &collection.getRawBuffer().front(),
	 TypeToCLType<typename THead::data_type>::cl_type * collection.size());

	 dataitems_tail::toDevice (
	 }

	 void fromDevice(dataitems_type & collection)
	 {
	 m_context.transfer_from_buffer(m_buffer,
	 &collection.getRawBuffer().front(),
	 TypeToCLType<typename THead::data_type>::cl_type * collection.size());

	 }*/

private:
	cl_mem m_buffer;
};

template<>
class OpenCLTransfer<> : private boost::noncopyable
{
public:
	void initBuffers(clever::context&, dataitems<> const&)
	{
	}
};

typedef OpenCLTransfer<GlobalX, GlobalY> HitCollectionTransfer;

class Hit: private CollectionView<HitCollection>
{
public:
// get a pointer to one hit in the collection
	Hit(HitCollection & collection, index_type i) :
			CollectionView<HitCollection>(collection, i)
	{

	}

// create a new hit in the collection
	Hit(HitCollection & collection) :
			CollectionView<HitCollection>(collection)
	{
	}

	float globalX() const
	{
		return getValue<GlobalX>();
	}

	float globalY() const
	{
		return getValue<GlobalY>();
	}

};

