// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: Event.proto

#ifndef PROTOBUF_Event_2eproto__INCLUDED
#define PROTOBUF_Event_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2003000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2003000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_reflection.h>
// @@protoc_insertion_point(includes)

namespace PEvent {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_Event_2eproto();
void protobuf_AssignDesc_Event_2eproto();
void protobuf_ShutdownFile_Event_2eproto();

class PGlobalPoint;
class PHit;
class PEvent;
class PEventContainer;

// ===================================================================

class PGlobalPoint : public ::google::protobuf::Message {
 public:
  PGlobalPoint();
  virtual ~PGlobalPoint();
  
  PGlobalPoint(const PGlobalPoint& from);
  
  inline PGlobalPoint& operator=(const PGlobalPoint& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const PGlobalPoint& default_instance();
  
  void Swap(PGlobalPoint* other);
  
  // implements Message ----------------------------------------------
  
  PGlobalPoint* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const PGlobalPoint& from);
  void MergeFrom(const PGlobalPoint& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // required float x = 1;
  inline bool has_x() const;
  inline void clear_x();
  static const int kXFieldNumber = 1;
  inline float x() const;
  inline void set_x(float value);
  
  // required float y = 2;
  inline bool has_y() const;
  inline void clear_y();
  static const int kYFieldNumber = 2;
  inline float y() const;
  inline void set_y(float value);
  
  // required float z = 3;
  inline bool has_z() const;
  inline void clear_z();
  static const int kZFieldNumber = 3;
  inline float z() const;
  inline void set_z(float value);
  
  // @@protoc_insertion_point(class_scope:PEvent.PGlobalPoint)
 private:
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  mutable int _cached_size_;
  
  float x_;
  float y_;
  float z_;
  friend void  protobuf_AddDesc_Event_2eproto();
  friend void protobuf_AssignDesc_Event_2eproto();
  friend void protobuf_ShutdownFile_Event_2eproto();
  
  ::google::protobuf::uint32 _has_bits_[(3 + 31) / 32];
  
  // WHY DOES & HAVE LOWER PRECEDENCE THAN != !?
  inline bool _has_bit(int index) const {
    return (_has_bits_[index / 32] & (1u << (index % 32))) != 0;
  }
  inline void _set_bit(int index) {
    _has_bits_[index / 32] |= (1u << (index % 32));
  }
  inline void _clear_bit(int index) {
    _has_bits_[index / 32] &= ~(1u << (index % 32));
  }
  
  void InitAsDefaultInstance();
  static PGlobalPoint* default_instance_;
};
// -------------------------------------------------------------------

class PHit : public ::google::protobuf::Message {
 public:
  PHit();
  virtual ~PHit();
  
  PHit(const PHit& from);
  
  inline PHit& operator=(const PHit& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const PHit& default_instance();
  
  void Swap(PHit* other);
  
  // implements Message ----------------------------------------------
  
  PHit* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const PHit& from);
  void MergeFrom(const PHit& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // required .PEvent.PGlobalPoint position = 1;
  inline bool has_position() const;
  inline void clear_position();
  static const int kPositionFieldNumber = 1;
  inline const ::PEvent::PGlobalPoint& position() const;
  inline ::PEvent::PGlobalPoint* mutable_position();
  
  // required uint64 layer = 2;
  inline bool has_layer() const;
  inline void clear_layer();
  static const int kLayerFieldNumber = 2;
  inline ::google::protobuf::uint64 layer() const;
  inline void set_layer(::google::protobuf::uint64 value);
  
  // required uint64 detectorId = 3;
  inline bool has_detectorid() const;
  inline void clear_detectorid();
  static const int kDetectorIdFieldNumber = 3;
  inline ::google::protobuf::uint64 detectorid() const;
  inline void set_detectorid(::google::protobuf::uint64 value);
  
  // required uint64 hitId = 4;
  inline bool has_hitid() const;
  inline void clear_hitid();
  static const int kHitIdFieldNumber = 4;
  inline ::google::protobuf::uint64 hitid() const;
  inline void set_hitid(::google::protobuf::uint64 value);
  
  // @@protoc_insertion_point(class_scope:PEvent.PHit)
 private:
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  mutable int _cached_size_;
  
  ::PEvent::PGlobalPoint* position_;
  ::google::protobuf::uint64 layer_;
  ::google::protobuf::uint64 detectorid_;
  ::google::protobuf::uint64 hitid_;
  friend void  protobuf_AddDesc_Event_2eproto();
  friend void protobuf_AssignDesc_Event_2eproto();
  friend void protobuf_ShutdownFile_Event_2eproto();
  
  ::google::protobuf::uint32 _has_bits_[(4 + 31) / 32];
  
  // WHY DOES & HAVE LOWER PRECEDENCE THAN != !?
  inline bool _has_bit(int index) const {
    return (_has_bits_[index / 32] & (1u << (index % 32))) != 0;
  }
  inline void _set_bit(int index) {
    _has_bits_[index / 32] |= (1u << (index % 32));
  }
  inline void _clear_bit(int index) {
    _has_bits_[index / 32] &= ~(1u << (index % 32));
  }
  
  void InitAsDefaultInstance();
  static PHit* default_instance_;
};
// -------------------------------------------------------------------

class PEvent : public ::google::protobuf::Message {
 public:
  PEvent();
  virtual ~PEvent();
  
  PEvent(const PEvent& from);
  
  inline PEvent& operator=(const PEvent& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const PEvent& default_instance();
  
  void Swap(PEvent* other);
  
  // implements Message ----------------------------------------------
  
  PEvent* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const PEvent& from);
  void MergeFrom(const PEvent& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // required uint64 runNumber = 1;
  inline bool has_runnumber() const;
  inline void clear_runnumber();
  static const int kRunNumberFieldNumber = 1;
  inline ::google::protobuf::uint64 runnumber() const;
  inline void set_runnumber(::google::protobuf::uint64 value);
  
  // required uint64 lumiSection = 2;
  inline bool has_lumisection() const;
  inline void clear_lumisection();
  static const int kLumiSectionFieldNumber = 2;
  inline ::google::protobuf::uint64 lumisection() const;
  inline void set_lumisection(::google::protobuf::uint64 value);
  
  // required uint64 eventNumber = 3;
  inline bool has_eventnumber() const;
  inline void clear_eventnumber();
  static const int kEventNumberFieldNumber = 3;
  inline ::google::protobuf::uint64 eventnumber() const;
  inline void set_eventnumber(::google::protobuf::uint64 value);
  
  // repeated .PEvent.PHit hits = 4;
  inline int hits_size() const;
  inline void clear_hits();
  static const int kHitsFieldNumber = 4;
  inline const ::PEvent::PHit& hits(int index) const;
  inline ::PEvent::PHit* mutable_hits(int index);
  inline ::PEvent::PHit* add_hits();
  inline const ::google::protobuf::RepeatedPtrField< ::PEvent::PHit >&
      hits() const;
  inline ::google::protobuf::RepeatedPtrField< ::PEvent::PHit >*
      mutable_hits();
  
  // @@protoc_insertion_point(class_scope:PEvent.PEvent)
 private:
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  mutable int _cached_size_;
  
  ::google::protobuf::uint64 runnumber_;
  ::google::protobuf::uint64 lumisection_;
  ::google::protobuf::uint64 eventnumber_;
  ::google::protobuf::RepeatedPtrField< ::PEvent::PHit > hits_;
  friend void  protobuf_AddDesc_Event_2eproto();
  friend void protobuf_AssignDesc_Event_2eproto();
  friend void protobuf_ShutdownFile_Event_2eproto();
  
  ::google::protobuf::uint32 _has_bits_[(4 + 31) / 32];
  
  // WHY DOES & HAVE LOWER PRECEDENCE THAN != !?
  inline bool _has_bit(int index) const {
    return (_has_bits_[index / 32] & (1u << (index % 32))) != 0;
  }
  inline void _set_bit(int index) {
    _has_bits_[index / 32] |= (1u << (index % 32));
  }
  inline void _clear_bit(int index) {
    _has_bits_[index / 32] &= ~(1u << (index % 32));
  }
  
  void InitAsDefaultInstance();
  static PEvent* default_instance_;
};
// -------------------------------------------------------------------

class PEventContainer : public ::google::protobuf::Message {
 public:
  PEventContainer();
  virtual ~PEventContainer();
  
  PEventContainer(const PEventContainer& from);
  
  inline PEventContainer& operator=(const PEventContainer& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const PEventContainer& default_instance();
  
  void Swap(PEventContainer* other);
  
  // implements Message ----------------------------------------------
  
  PEventContainer* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const PEventContainer& from);
  void MergeFrom(const PEventContainer& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // repeated .PEvent.PEvent events = 1;
  inline int events_size() const;
  inline void clear_events();
  static const int kEventsFieldNumber = 1;
  inline const ::PEvent::PEvent& events(int index) const;
  inline ::PEvent::PEvent* mutable_events(int index);
  inline ::PEvent::PEvent* add_events();
  inline const ::google::protobuf::RepeatedPtrField< ::PEvent::PEvent >&
      events() const;
  inline ::google::protobuf::RepeatedPtrField< ::PEvent::PEvent >*
      mutable_events();
  
  // @@protoc_insertion_point(class_scope:PEvent.PEventContainer)
 private:
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  mutable int _cached_size_;
  
  ::google::protobuf::RepeatedPtrField< ::PEvent::PEvent > events_;
  friend void  protobuf_AddDesc_Event_2eproto();
  friend void protobuf_AssignDesc_Event_2eproto();
  friend void protobuf_ShutdownFile_Event_2eproto();
  
  ::google::protobuf::uint32 _has_bits_[(1 + 31) / 32];
  
  // WHY DOES & HAVE LOWER PRECEDENCE THAN != !?
  inline bool _has_bit(int index) const {
    return (_has_bits_[index / 32] & (1u << (index % 32))) != 0;
  }
  inline void _set_bit(int index) {
    _has_bits_[index / 32] |= (1u << (index % 32));
  }
  inline void _clear_bit(int index) {
    _has_bits_[index / 32] &= ~(1u << (index % 32));
  }
  
  void InitAsDefaultInstance();
  static PEventContainer* default_instance_;
};
// ===================================================================


// ===================================================================

// PGlobalPoint

// required float x = 1;
inline bool PGlobalPoint::has_x() const {
  return _has_bit(0);
}
inline void PGlobalPoint::clear_x() {
  x_ = 0;
  _clear_bit(0);
}
inline float PGlobalPoint::x() const {
  return x_;
}
inline void PGlobalPoint::set_x(float value) {
  _set_bit(0);
  x_ = value;
}

// required float y = 2;
inline bool PGlobalPoint::has_y() const {
  return _has_bit(1);
}
inline void PGlobalPoint::clear_y() {
  y_ = 0;
  _clear_bit(1);
}
inline float PGlobalPoint::y() const {
  return y_;
}
inline void PGlobalPoint::set_y(float value) {
  _set_bit(1);
  y_ = value;
}

// required float z = 3;
inline bool PGlobalPoint::has_z() const {
  return _has_bit(2);
}
inline void PGlobalPoint::clear_z() {
  z_ = 0;
  _clear_bit(2);
}
inline float PGlobalPoint::z() const {
  return z_;
}
inline void PGlobalPoint::set_z(float value) {
  _set_bit(2);
  z_ = value;
}

// -------------------------------------------------------------------

// PHit

// required .PEvent.PGlobalPoint position = 1;
inline bool PHit::has_position() const {
  return _has_bit(0);
}
inline void PHit::clear_position() {
  if (position_ != NULL) position_->::PEvent::PGlobalPoint::Clear();
  _clear_bit(0);
}
inline const ::PEvent::PGlobalPoint& PHit::position() const {
  return position_ != NULL ? *position_ : *default_instance_->position_;
}
inline ::PEvent::PGlobalPoint* PHit::mutable_position() {
  _set_bit(0);
  if (position_ == NULL) position_ = new ::PEvent::PGlobalPoint;
  return position_;
}

// required uint64 layer = 2;
inline bool PHit::has_layer() const {
  return _has_bit(1);
}
inline void PHit::clear_layer() {
  layer_ = GOOGLE_ULONGLONG(0);
  _clear_bit(1);
}
inline ::google::protobuf::uint64 PHit::layer() const {
  return layer_;
}
inline void PHit::set_layer(::google::protobuf::uint64 value) {
  _set_bit(1);
  layer_ = value;
}

// required uint64 detectorId = 3;
inline bool PHit::has_detectorid() const {
  return _has_bit(2);
}
inline void PHit::clear_detectorid() {
  detectorid_ = GOOGLE_ULONGLONG(0);
  _clear_bit(2);
}
inline ::google::protobuf::uint64 PHit::detectorid() const {
  return detectorid_;
}
inline void PHit::set_detectorid(::google::protobuf::uint64 value) {
  _set_bit(2);
  detectorid_ = value;
}

// required uint64 hitId = 4;
inline bool PHit::has_hitid() const {
  return _has_bit(3);
}
inline void PHit::clear_hitid() {
  hitid_ = GOOGLE_ULONGLONG(0);
  _clear_bit(3);
}
inline ::google::protobuf::uint64 PHit::hitid() const {
  return hitid_;
}
inline void PHit::set_hitid(::google::protobuf::uint64 value) {
  _set_bit(3);
  hitid_ = value;
}

// -------------------------------------------------------------------

// PEvent

// required uint64 runNumber = 1;
inline bool PEvent::has_runnumber() const {
  return _has_bit(0);
}
inline void PEvent::clear_runnumber() {
  runnumber_ = GOOGLE_ULONGLONG(0);
  _clear_bit(0);
}
inline ::google::protobuf::uint64 PEvent::runnumber() const {
  return runnumber_;
}
inline void PEvent::set_runnumber(::google::protobuf::uint64 value) {
  _set_bit(0);
  runnumber_ = value;
}

// required uint64 lumiSection = 2;
inline bool PEvent::has_lumisection() const {
  return _has_bit(1);
}
inline void PEvent::clear_lumisection() {
  lumisection_ = GOOGLE_ULONGLONG(0);
  _clear_bit(1);
}
inline ::google::protobuf::uint64 PEvent::lumisection() const {
  return lumisection_;
}
inline void PEvent::set_lumisection(::google::protobuf::uint64 value) {
  _set_bit(1);
  lumisection_ = value;
}

// required uint64 eventNumber = 3;
inline bool PEvent::has_eventnumber() const {
  return _has_bit(2);
}
inline void PEvent::clear_eventnumber() {
  eventnumber_ = GOOGLE_ULONGLONG(0);
  _clear_bit(2);
}
inline ::google::protobuf::uint64 PEvent::eventnumber() const {
  return eventnumber_;
}
inline void PEvent::set_eventnumber(::google::protobuf::uint64 value) {
  _set_bit(2);
  eventnumber_ = value;
}

// repeated .PEvent.PHit hits = 4;
inline int PEvent::hits_size() const {
  return hits_.size();
}
inline void PEvent::clear_hits() {
  hits_.Clear();
}
inline const ::PEvent::PHit& PEvent::hits(int index) const {
  return hits_.Get(index);
}
inline ::PEvent::PHit* PEvent::mutable_hits(int index) {
  return hits_.Mutable(index);
}
inline ::PEvent::PHit* PEvent::add_hits() {
  return hits_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::PEvent::PHit >&
PEvent::hits() const {
  return hits_;
}
inline ::google::protobuf::RepeatedPtrField< ::PEvent::PHit >*
PEvent::mutable_hits() {
  return &hits_;
}

// -------------------------------------------------------------------

// PEventContainer

// repeated .PEvent.PEvent events = 1;
inline int PEventContainer::events_size() const {
  return events_.size();
}
inline void PEventContainer::clear_events() {
  events_.Clear();
}
inline const ::PEvent::PEvent& PEventContainer::events(int index) const {
  return events_.Get(index);
}
inline ::PEvent::PEvent* PEventContainer::mutable_events(int index) {
  return events_.Mutable(index);
}
inline ::PEvent::PEvent* PEventContainer::add_events() {
  return events_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::PEvent::PEvent >&
PEventContainer::events() const {
  return events_;
}
inline ::google::protobuf::RepeatedPtrField< ::PEvent::PEvent >*
PEventContainer::mutable_events() {
  return &events_;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace PEvent

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_Event_2eproto__INCLUDED
