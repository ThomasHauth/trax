// Generated by the protocol buffer compiler.  DO NOT EDIT!

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "Event.pb.h"
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace PEvent {

namespace {

const ::google::protobuf::Descriptor* PGlobalPoint_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  PGlobalPoint_reflection_ = NULL;
const ::google::protobuf::Descriptor* PHit_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  PHit_reflection_ = NULL;
const ::google::protobuf::Descriptor* PEvent_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  PEvent_reflection_ = NULL;
const ::google::protobuf::Descriptor* PEventContainer_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  PEventContainer_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_Event_2eproto() {
  protobuf_AddDesc_Event_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "Event.proto");
  GOOGLE_CHECK(file != NULL);
  PGlobalPoint_descriptor_ = file->message_type(0);
  static const int PGlobalPoint_offsets_[3] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PGlobalPoint, x_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PGlobalPoint, y_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PGlobalPoint, z_),
  };
  PGlobalPoint_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      PGlobalPoint_descriptor_,
      PGlobalPoint::default_instance_,
      PGlobalPoint_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PGlobalPoint, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PGlobalPoint, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(PGlobalPoint));
  PHit_descriptor_ = file->message_type(1);
  static const int PHit_offsets_[3] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PHit, position_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PHit, layer_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PHit, detectorid_),
  };
  PHit_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      PHit_descriptor_,
      PHit::default_instance_,
      PHit_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PHit, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PHit, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(PHit));
  PEvent_descriptor_ = file->message_type(2);
  static const int PEvent_offsets_[4] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEvent, runnumber_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEvent, lumisection_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEvent, eventnumber_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEvent, hits_),
  };
  PEvent_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      PEvent_descriptor_,
      PEvent::default_instance_,
      PEvent_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEvent, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEvent, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(PEvent));
  PEventContainer_descriptor_ = file->message_type(3);
  static const int PEventContainer_offsets_[1] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEventContainer, events_),
  };
  PEventContainer_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      PEventContainer_descriptor_,
      PEventContainer::default_instance_,
      PEventContainer_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEventContainer, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PEventContainer, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(PEventContainer));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_Event_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    PGlobalPoint_descriptor_, &PGlobalPoint::default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    PHit_descriptor_, &PHit::default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    PEvent_descriptor_, &PEvent::default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    PEventContainer_descriptor_, &PEventContainer::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_Event_2eproto() {
  delete PGlobalPoint::default_instance_;
  delete PGlobalPoint_reflection_;
  delete PHit::default_instance_;
  delete PHit_reflection_;
  delete PEvent::default_instance_;
  delete PEvent_reflection_;
  delete PEventContainer::default_instance_;
  delete PEventContainer_reflection_;
}

void protobuf_AddDesc_Event_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\013Event.proto\022\006PEvent\"/\n\014PGlobalPoint\022\t\n"
    "\001x\030\001 \002(\002\022\t\n\001y\030\002 \002(\002\022\t\n\001z\030\003 \002(\002\"Q\n\004PHit\022&"
    "\n\010position\030\001 \002(\0132\024.PEvent.PGlobalPoint\022\r"
    "\n\005layer\030\002 \002(\004\022\022\n\ndetectorId\030\003 \002(\004\"a\n\006PEv"
    "ent\022\021\n\trunNumber\030\001 \002(\004\022\023\n\013lumiSection\030\002 "
    "\002(\004\022\023\n\013eventNumber\030\003 \002(\004\022\032\n\004hits\030\004 \003(\0132\014"
    ".PEvent.PHit\"1\n\017PEventContainer\022\036\n\006event"
    "s\030\001 \003(\0132\016.PEvent.PEvent", 303);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "Event.proto", &protobuf_RegisterTypes);
  PGlobalPoint::default_instance_ = new PGlobalPoint();
  PHit::default_instance_ = new PHit();
  PEvent::default_instance_ = new PEvent();
  PEventContainer::default_instance_ = new PEventContainer();
  PGlobalPoint::default_instance_->InitAsDefaultInstance();
  PHit::default_instance_->InitAsDefaultInstance();
  PEvent::default_instance_->InitAsDefaultInstance();
  PEventContainer::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_Event_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_Event_2eproto {
  StaticDescriptorInitializer_Event_2eproto() {
    protobuf_AddDesc_Event_2eproto();
  }
} static_descriptor_initializer_Event_2eproto_;


// ===================================================================

#ifndef _MSC_VER
const int PGlobalPoint::kXFieldNumber;
const int PGlobalPoint::kYFieldNumber;
const int PGlobalPoint::kZFieldNumber;
#endif  // !_MSC_VER

PGlobalPoint::PGlobalPoint()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void PGlobalPoint::InitAsDefaultInstance() {
}

PGlobalPoint::PGlobalPoint(const PGlobalPoint& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void PGlobalPoint::SharedCtor() {
  _cached_size_ = 0;
  x_ = 0;
  y_ = 0;
  z_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

PGlobalPoint::~PGlobalPoint() {
  SharedDtor();
}

void PGlobalPoint::SharedDtor() {
  if (this != default_instance_) {
  }
}

void PGlobalPoint::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* PGlobalPoint::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return PGlobalPoint_descriptor_;
}

const PGlobalPoint& PGlobalPoint::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_Event_2eproto();  return *default_instance_;
}

PGlobalPoint* PGlobalPoint::default_instance_ = NULL;

PGlobalPoint* PGlobalPoint::New() const {
  return new PGlobalPoint;
}

void PGlobalPoint::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    x_ = 0;
    y_ = 0;
    z_ = 0;
  }
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool PGlobalPoint::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required float x = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED32) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &x_)));
          _set_bit(0);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(21)) goto parse_y;
        break;
      }
      
      // required float y = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED32) {
         parse_y:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &y_)));
          _set_bit(1);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(29)) goto parse_z;
        break;
      }
      
      // required float z = 3;
      case 3: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED32) {
         parse_z:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &z_)));
          _set_bit(2);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void PGlobalPoint::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // required float x = 1;
  if (_has_bit(0)) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(1, this->x(), output);
  }
  
  // required float y = 2;
  if (_has_bit(1)) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(2, this->y(), output);
  }
  
  // required float z = 3;
  if (_has_bit(2)) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(3, this->z(), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* PGlobalPoint::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // required float x = 1;
  if (_has_bit(0)) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(1, this->x(), target);
  }
  
  // required float y = 2;
  if (_has_bit(1)) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(2, this->y(), target);
  }
  
  // required float z = 3;
  if (_has_bit(2)) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(3, this->z(), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int PGlobalPoint::ByteSize() const {
  int total_size = 0;
  
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required float x = 1;
    if (has_x()) {
      total_size += 1 + 4;
    }
    
    // required float y = 2;
    if (has_y()) {
      total_size += 1 + 4;
    }
    
    // required float z = 3;
    if (has_z()) {
      total_size += 1 + 4;
    }
    
  }
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void PGlobalPoint::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const PGlobalPoint* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const PGlobalPoint*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void PGlobalPoint::MergeFrom(const PGlobalPoint& from) {
  GOOGLE_CHECK_NE(&from, this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from._has_bit(0)) {
      set_x(from.x());
    }
    if (from._has_bit(1)) {
      set_y(from.y());
    }
    if (from._has_bit(2)) {
      set_z(from.z());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void PGlobalPoint::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void PGlobalPoint::CopyFrom(const PGlobalPoint& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool PGlobalPoint::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000007) != 0x00000007) return false;
  
  return true;
}

void PGlobalPoint::Swap(PGlobalPoint* other) {
  if (other != this) {
    std::swap(x_, other->x_);
    std::swap(y_, other->y_);
    std::swap(z_, other->z_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata PGlobalPoint::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = PGlobalPoint_descriptor_;
  metadata.reflection = PGlobalPoint_reflection_;
  return metadata;
}


// ===================================================================

#ifndef _MSC_VER
const int PHit::kPositionFieldNumber;
const int PHit::kLayerFieldNumber;
const int PHit::kDetectorIdFieldNumber;
#endif  // !_MSC_VER

PHit::PHit()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void PHit::InitAsDefaultInstance() {
  position_ = const_cast< ::PEvent::PGlobalPoint*>(&::PEvent::PGlobalPoint::default_instance());
}

PHit::PHit(const PHit& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void PHit::SharedCtor() {
  _cached_size_ = 0;
  position_ = NULL;
  layer_ = GOOGLE_ULONGLONG(0);
  detectorid_ = GOOGLE_ULONGLONG(0);
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

PHit::~PHit() {
  SharedDtor();
}

void PHit::SharedDtor() {
  if (this != default_instance_) {
    delete position_;
  }
}

void PHit::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* PHit::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return PHit_descriptor_;
}

const PHit& PHit::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_Event_2eproto();  return *default_instance_;
}

PHit* PHit::default_instance_ = NULL;

PHit* PHit::New() const {
  return new PHit;
}

void PHit::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (_has_bit(0)) {
      if (position_ != NULL) position_->::PEvent::PGlobalPoint::Clear();
    }
    layer_ = GOOGLE_ULONGLONG(0);
    detectorid_ = GOOGLE_ULONGLONG(0);
  }
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool PHit::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required .PEvent.PGlobalPoint position = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_position()));
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(16)) goto parse_layer;
        break;
      }
      
      // required uint64 layer = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_layer:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &layer_)));
          _set_bit(1);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(24)) goto parse_detectorId;
        break;
      }
      
      // required uint64 detectorId = 3;
      case 3: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_detectorId:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &detectorid_)));
          _set_bit(2);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void PHit::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // required .PEvent.PGlobalPoint position = 1;
  if (_has_bit(0)) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->position(), output);
  }
  
  // required uint64 layer = 2;
  if (_has_bit(1)) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(2, this->layer(), output);
  }
  
  // required uint64 detectorId = 3;
  if (_has_bit(2)) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(3, this->detectorid(), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* PHit::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // required .PEvent.PGlobalPoint position = 1;
  if (_has_bit(0)) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        1, this->position(), target);
  }
  
  // required uint64 layer = 2;
  if (_has_bit(1)) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(2, this->layer(), target);
  }
  
  // required uint64 detectorId = 3;
  if (_has_bit(2)) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(3, this->detectorid(), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int PHit::ByteSize() const {
  int total_size = 0;
  
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required .PEvent.PGlobalPoint position = 1;
    if (has_position()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->position());
    }
    
    // required uint64 layer = 2;
    if (has_layer()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt64Size(
          this->layer());
    }
    
    // required uint64 detectorId = 3;
    if (has_detectorid()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt64Size(
          this->detectorid());
    }
    
  }
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void PHit::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const PHit* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const PHit*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void PHit::MergeFrom(const PHit& from) {
  GOOGLE_CHECK_NE(&from, this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from._has_bit(0)) {
      mutable_position()->::PEvent::PGlobalPoint::MergeFrom(from.position());
    }
    if (from._has_bit(1)) {
      set_layer(from.layer());
    }
    if (from._has_bit(2)) {
      set_detectorid(from.detectorid());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void PHit::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void PHit::CopyFrom(const PHit& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool PHit::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000007) != 0x00000007) return false;
  
  if (has_position()) {
    if (!this->position().IsInitialized()) return false;
  }
  return true;
}

void PHit::Swap(PHit* other) {
  if (other != this) {
    std::swap(position_, other->position_);
    std::swap(layer_, other->layer_);
    std::swap(detectorid_, other->detectorid_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata PHit::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = PHit_descriptor_;
  metadata.reflection = PHit_reflection_;
  return metadata;
}


// ===================================================================

#ifndef _MSC_VER
const int PEvent::kRunNumberFieldNumber;
const int PEvent::kLumiSectionFieldNumber;
const int PEvent::kEventNumberFieldNumber;
const int PEvent::kHitsFieldNumber;
#endif  // !_MSC_VER

PEvent::PEvent()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void PEvent::InitAsDefaultInstance() {
}

PEvent::PEvent(const PEvent& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void PEvent::SharedCtor() {
  _cached_size_ = 0;
  runnumber_ = GOOGLE_ULONGLONG(0);
  lumisection_ = GOOGLE_ULONGLONG(0);
  eventnumber_ = GOOGLE_ULONGLONG(0);
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

PEvent::~PEvent() {
  SharedDtor();
}

void PEvent::SharedDtor() {
  if (this != default_instance_) {
  }
}

void PEvent::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* PEvent::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return PEvent_descriptor_;
}

const PEvent& PEvent::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_Event_2eproto();  return *default_instance_;
}

PEvent* PEvent::default_instance_ = NULL;

PEvent* PEvent::New() const {
  return new PEvent;
}

void PEvent::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    runnumber_ = GOOGLE_ULONGLONG(0);
    lumisection_ = GOOGLE_ULONGLONG(0);
    eventnumber_ = GOOGLE_ULONGLONG(0);
  }
  hits_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool PEvent::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required uint64 runNumber = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &runnumber_)));
          _set_bit(0);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(16)) goto parse_lumiSection;
        break;
      }
      
      // required uint64 lumiSection = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_lumiSection:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &lumisection_)));
          _set_bit(1);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(24)) goto parse_eventNumber;
        break;
      }
      
      // required uint64 eventNumber = 3;
      case 3: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_eventNumber:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &eventnumber_)));
          _set_bit(2);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(34)) goto parse_hits;
        break;
      }
      
      // repeated .PEvent.PHit hits = 4;
      case 4: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
         parse_hits:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
                input, add_hits()));
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(34)) goto parse_hits;
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void PEvent::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // required uint64 runNumber = 1;
  if (_has_bit(0)) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(1, this->runnumber(), output);
  }
  
  // required uint64 lumiSection = 2;
  if (_has_bit(1)) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(2, this->lumisection(), output);
  }
  
  // required uint64 eventNumber = 3;
  if (_has_bit(2)) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(3, this->eventnumber(), output);
  }
  
  // repeated .PEvent.PHit hits = 4;
  for (int i = 0; i < this->hits_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      4, this->hits(i), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* PEvent::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // required uint64 runNumber = 1;
  if (_has_bit(0)) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(1, this->runnumber(), target);
  }
  
  // required uint64 lumiSection = 2;
  if (_has_bit(1)) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(2, this->lumisection(), target);
  }
  
  // required uint64 eventNumber = 3;
  if (_has_bit(2)) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(3, this->eventnumber(), target);
  }
  
  // repeated .PEvent.PHit hits = 4;
  for (int i = 0; i < this->hits_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        4, this->hits(i), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int PEvent::ByteSize() const {
  int total_size = 0;
  
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required uint64 runNumber = 1;
    if (has_runnumber()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt64Size(
          this->runnumber());
    }
    
    // required uint64 lumiSection = 2;
    if (has_lumisection()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt64Size(
          this->lumisection());
    }
    
    // required uint64 eventNumber = 3;
    if (has_eventnumber()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt64Size(
          this->eventnumber());
    }
    
  }
  // repeated .PEvent.PHit hits = 4;
  total_size += 1 * this->hits_size();
  for (int i = 0; i < this->hits_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->hits(i));
  }
  
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void PEvent::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const PEvent* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const PEvent*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void PEvent::MergeFrom(const PEvent& from) {
  GOOGLE_CHECK_NE(&from, this);
  hits_.MergeFrom(from.hits_);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from._has_bit(0)) {
      set_runnumber(from.runnumber());
    }
    if (from._has_bit(1)) {
      set_lumisection(from.lumisection());
    }
    if (from._has_bit(2)) {
      set_eventnumber(from.eventnumber());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void PEvent::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void PEvent::CopyFrom(const PEvent& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool PEvent::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000007) != 0x00000007) return false;
  
  for (int i = 0; i < hits_size(); i++) {
    if (!this->hits(i).IsInitialized()) return false;
  }
  return true;
}

void PEvent::Swap(PEvent* other) {
  if (other != this) {
    std::swap(runnumber_, other->runnumber_);
    std::swap(lumisection_, other->lumisection_);
    std::swap(eventnumber_, other->eventnumber_);
    hits_.Swap(&other->hits_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata PEvent::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = PEvent_descriptor_;
  metadata.reflection = PEvent_reflection_;
  return metadata;
}


// ===================================================================

#ifndef _MSC_VER
const int PEventContainer::kEventsFieldNumber;
#endif  // !_MSC_VER

PEventContainer::PEventContainer()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void PEventContainer::InitAsDefaultInstance() {
}

PEventContainer::PEventContainer(const PEventContainer& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void PEventContainer::SharedCtor() {
  _cached_size_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

PEventContainer::~PEventContainer() {
  SharedDtor();
}

void PEventContainer::SharedDtor() {
  if (this != default_instance_) {
  }
}

void PEventContainer::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* PEventContainer::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return PEventContainer_descriptor_;
}

const PEventContainer& PEventContainer::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_Event_2eproto();  return *default_instance_;
}

PEventContainer* PEventContainer::default_instance_ = NULL;

PEventContainer* PEventContainer::New() const {
  return new PEventContainer;
}

void PEventContainer::Clear() {
  events_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool PEventContainer::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .PEvent.PEvent events = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
         parse_events:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
                input, add_events()));
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(10)) goto parse_events;
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void PEventContainer::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // repeated .PEvent.PEvent events = 1;
  for (int i = 0; i < this->events_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->events(i), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* PEventContainer::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // repeated .PEvent.PEvent events = 1;
  for (int i = 0; i < this->events_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        1, this->events(i), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int PEventContainer::ByteSize() const {
  int total_size = 0;
  
  // repeated .PEvent.PEvent events = 1;
  total_size += 1 * this->events_size();
  for (int i = 0; i < this->events_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->events(i));
  }
  
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void PEventContainer::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const PEventContainer* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const PEventContainer*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void PEventContainer::MergeFrom(const PEventContainer& from) {
  GOOGLE_CHECK_NE(&from, this);
  events_.MergeFrom(from.events_);
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void PEventContainer::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void PEventContainer::CopyFrom(const PEventContainer& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool PEventContainer::IsInitialized() const {
  
  for (int i = 0; i < events_size(); i++) {
    if (!this->events(i).IsInitialized()) return false;
  }
  return true;
}

void PEventContainer::Swap(PEventContainer* other) {
  if (other != this) {
    events_.Swap(&other->events_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata PEventContainer::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = PEventContainer_descriptor_;
  metadata.reflection = PEventContainer_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace PEvent

// @@protoc_insertion_point(global_scope)
