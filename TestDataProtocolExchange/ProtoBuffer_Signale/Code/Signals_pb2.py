# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Signals.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Signals.proto',
  package='Signals',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rSignals.proto\x12\x07Signals\"\x1a\n\tBytesList\x12\r\n\x05value\x18\x01 \x03(\x0c\"\x1e\n\tFloatList\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"\x1e\n\tInt64List\x12\x11\n\x05value\x18\x01 \x03(\x03\x42\x02\x10\x01\"/\n\nBytesTuple\x12\r\n\x05value\x18\x01 \x01(\x0c\x12\x12\n\ntime_stamp\x18\x02 \x01(\x03\"/\n\nInt64Tuple\x12\r\n\x05value\x18\x01 \x01(\x03\x12\x12\n\ntime_stamp\x18\x02 \x01(\x03\"/\n\nFloatTuple\x12\r\n\x05value\x18\x01 \x01(\x02\x12\x12\n\ntime_stamp\x18\x02 \x01(\x03\"4\n\x0e\x42ytesTupleList\x12\"\n\x05value\x18\x01 \x03(\x0b\x32\x13.Signals.BytesTuple\"4\n\x0eInt64TupleList\x12\"\n\x05value\x18\x01 \x03(\x0b\x32\x13.Signals.Int64Tuple\"4\n\x0e\x46loatTupleList\x12\"\n\x05value\x18\x01 \x03(\x0b\x32\x13.Signals.FloatTuple\"\xae\x02\n\x07\x46\x65\x61ture\x12(\n\nbytes_list\x18\x01 \x01(\x0b\x32\x12.Signals.BytesListH\x00\x12(\n\nfloat_list\x18\x02 \x01(\x0b\x32\x12.Signals.FloatListH\x00\x12(\n\nint64_list\x18\x03 \x01(\x0b\x32\x12.Signals.Int64ListH\x00\x12\x33\n\x10\x62ytes_tuple_list\x18\x04 \x01(\x0b\x32\x17.Signals.BytesTupleListH\x00\x12\x33\n\x10int64_tuple_list\x18\x05 \x01(\x0b\x32\x17.Signals.Int64TupleListH\x00\x12\x33\n\x10\x66loat_tuple_list\x18\x06 \x01(\x0b\x32\x17.Signals.FloatTupleListH\x00\x42\x06\n\x04kind\"}\n\x08\x46\x65\x61tures\x12/\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32\x1e.Signals.Features.FeatureEntry\x1a@\n\x0c\x46\x65\x61tureEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1f\n\x05value\x18\x02 \x01(\x0b\x32\x10.Signals.Feature:\x02\x38\x01\".\n\x07\x45xample\x12#\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x11.Signals.Features\"0\n\x0b\x46\x65\x61tureList\x12!\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32\x10.Signals.Feature\"\x96\x01\n\x0c\x46\x65\x61tureLists\x12<\n\x0c\x66\x65\x61ture_list\x18\x01 \x03(\x0b\x32&.Signals.FeatureLists.FeatureListEntry\x1aH\n\x10\x46\x65\x61tureListEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.Signals.FeatureList:\x02\x38\x01\"c\n\x0fSequenceExample\x12\"\n\x07\x63ontext\x18\x01 \x01(\x0b\x32\x11.Signals.Features\x12,\n\rfeature_lists\x18\x02 \x01(\x0b\x32\x15.Signals.FeatureListsb\x06proto3')
)




_BYTESLIST = _descriptor.Descriptor(
  name='BytesList',
  full_name='Signals.BytesList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.BytesList.value', index=0,
      number=1, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26,
  serialized_end=52,
)


_FLOATLIST = _descriptor.Descriptor(
  name='FloatList',
  full_name='Signals.FloatList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.FloatList.value', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=54,
  serialized_end=84,
)


_INT64LIST = _descriptor.Descriptor(
  name='Int64List',
  full_name='Signals.Int64List',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.Int64List.value', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=86,
  serialized_end=116,
)


_BYTESTUPLE = _descriptor.Descriptor(
  name='BytesTuple',
  full_name='Signals.BytesTuple',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.BytesTuple.value', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time_stamp', full_name='Signals.BytesTuple.time_stamp', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=118,
  serialized_end=165,
)


_INT64TUPLE = _descriptor.Descriptor(
  name='Int64Tuple',
  full_name='Signals.Int64Tuple',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.Int64Tuple.value', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time_stamp', full_name='Signals.Int64Tuple.time_stamp', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=167,
  serialized_end=214,
)


_FLOATTUPLE = _descriptor.Descriptor(
  name='FloatTuple',
  full_name='Signals.FloatTuple',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.FloatTuple.value', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time_stamp', full_name='Signals.FloatTuple.time_stamp', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=216,
  serialized_end=263,
)


_BYTESTUPLELIST = _descriptor.Descriptor(
  name='BytesTupleList',
  full_name='Signals.BytesTupleList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.BytesTupleList.value', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=265,
  serialized_end=317,
)


_INT64TUPLELIST = _descriptor.Descriptor(
  name='Int64TupleList',
  full_name='Signals.Int64TupleList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.Int64TupleList.value', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=319,
  serialized_end=371,
)


_FLOATTUPLELIST = _descriptor.Descriptor(
  name='FloatTupleList',
  full_name='Signals.FloatTupleList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.FloatTupleList.value', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=373,
  serialized_end=425,
)


_FEATURE = _descriptor.Descriptor(
  name='Feature',
  full_name='Signals.Feature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bytes_list', full_name='Signals.Feature.bytes_list', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='float_list', full_name='Signals.Feature.float_list', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int64_list', full_name='Signals.Feature.int64_list', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bytes_tuple_list', full_name='Signals.Feature.bytes_tuple_list', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int64_tuple_list', full_name='Signals.Feature.int64_tuple_list', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='float_tuple_list', full_name='Signals.Feature.float_tuple_list', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='kind', full_name='Signals.Feature.kind',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=428,
  serialized_end=730,
)


_FEATURES_FEATUREENTRY = _descriptor.Descriptor(
  name='FeatureEntry',
  full_name='Signals.Features.FeatureEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='Signals.Features.FeatureEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.Features.FeatureEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=793,
  serialized_end=857,
)

_FEATURES = _descriptor.Descriptor(
  name='Features',
  full_name='Signals.Features',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature', full_name='Signals.Features.feature', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FEATURES_FEATUREENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=732,
  serialized_end=857,
)


_EXAMPLE = _descriptor.Descriptor(
  name='Example',
  full_name='Signals.Example',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='features', full_name='Signals.Example.features', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=859,
  serialized_end=905,
)


_FEATURELIST = _descriptor.Descriptor(
  name='FeatureList',
  full_name='Signals.FeatureList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature', full_name='Signals.FeatureList.feature', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=907,
  serialized_end=955,
)


_FEATURELISTS_FEATURELISTENTRY = _descriptor.Descriptor(
  name='FeatureListEntry',
  full_name='Signals.FeatureLists.FeatureListEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='Signals.FeatureLists.FeatureListEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='Signals.FeatureLists.FeatureListEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1036,
  serialized_end=1108,
)

_FEATURELISTS = _descriptor.Descriptor(
  name='FeatureLists',
  full_name='Signals.FeatureLists',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature_list', full_name='Signals.FeatureLists.feature_list', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FEATURELISTS_FEATURELISTENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=958,
  serialized_end=1108,
)


_SEQUENCEEXAMPLE = _descriptor.Descriptor(
  name='SequenceExample',
  full_name='Signals.SequenceExample',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='Signals.SequenceExample.context', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_lists', full_name='Signals.SequenceExample.feature_lists', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1110,
  serialized_end=1209,
)

_BYTESTUPLELIST.fields_by_name['value'].message_type = _BYTESTUPLE
_INT64TUPLELIST.fields_by_name['value'].message_type = _INT64TUPLE
_FLOATTUPLELIST.fields_by_name['value'].message_type = _FLOATTUPLE
_FEATURE.fields_by_name['bytes_list'].message_type = _BYTESLIST
_FEATURE.fields_by_name['float_list'].message_type = _FLOATLIST
_FEATURE.fields_by_name['int64_list'].message_type = _INT64LIST
_FEATURE.fields_by_name['bytes_tuple_list'].message_type = _BYTESTUPLELIST
_FEATURE.fields_by_name['int64_tuple_list'].message_type = _INT64TUPLELIST
_FEATURE.fields_by_name['float_tuple_list'].message_type = _FLOATTUPLELIST
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['bytes_list'])
_FEATURE.fields_by_name['bytes_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['float_list'])
_FEATURE.fields_by_name['float_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['int64_list'])
_FEATURE.fields_by_name['int64_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['bytes_tuple_list'])
_FEATURE.fields_by_name['bytes_tuple_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['int64_tuple_list'])
_FEATURE.fields_by_name['int64_tuple_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['float_tuple_list'])
_FEATURE.fields_by_name['float_tuple_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURES_FEATUREENTRY.fields_by_name['value'].message_type = _FEATURE
_FEATURES_FEATUREENTRY.containing_type = _FEATURES
_FEATURES.fields_by_name['feature'].message_type = _FEATURES_FEATUREENTRY
_EXAMPLE.fields_by_name['features'].message_type = _FEATURES
_FEATURELIST.fields_by_name['feature'].message_type = _FEATURE
_FEATURELISTS_FEATURELISTENTRY.fields_by_name['value'].message_type = _FEATURELIST
_FEATURELISTS_FEATURELISTENTRY.containing_type = _FEATURELISTS
_FEATURELISTS.fields_by_name['feature_list'].message_type = _FEATURELISTS_FEATURELISTENTRY
_SEQUENCEEXAMPLE.fields_by_name['context'].message_type = _FEATURES
_SEQUENCEEXAMPLE.fields_by_name['feature_lists'].message_type = _FEATURELISTS
DESCRIPTOR.message_types_by_name['BytesList'] = _BYTESLIST
DESCRIPTOR.message_types_by_name['FloatList'] = _FLOATLIST
DESCRIPTOR.message_types_by_name['Int64List'] = _INT64LIST
DESCRIPTOR.message_types_by_name['BytesTuple'] = _BYTESTUPLE
DESCRIPTOR.message_types_by_name['Int64Tuple'] = _INT64TUPLE
DESCRIPTOR.message_types_by_name['FloatTuple'] = _FLOATTUPLE
DESCRIPTOR.message_types_by_name['BytesTupleList'] = _BYTESTUPLELIST
DESCRIPTOR.message_types_by_name['Int64TupleList'] = _INT64TUPLELIST
DESCRIPTOR.message_types_by_name['FloatTupleList'] = _FLOATTUPLELIST
DESCRIPTOR.message_types_by_name['Feature'] = _FEATURE
DESCRIPTOR.message_types_by_name['Features'] = _FEATURES
DESCRIPTOR.message_types_by_name['Example'] = _EXAMPLE
DESCRIPTOR.message_types_by_name['FeatureList'] = _FEATURELIST
DESCRIPTOR.message_types_by_name['FeatureLists'] = _FEATURELISTS
DESCRIPTOR.message_types_by_name['SequenceExample'] = _SEQUENCEEXAMPLE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BytesList = _reflection.GeneratedProtocolMessageType('BytesList', (_message.Message,), dict(
  DESCRIPTOR = _BYTESLIST,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.BytesList)
  ))
_sym_db.RegisterMessage(BytesList)

FloatList = _reflection.GeneratedProtocolMessageType('FloatList', (_message.Message,), dict(
  DESCRIPTOR = _FLOATLIST,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.FloatList)
  ))
_sym_db.RegisterMessage(FloatList)

Int64List = _reflection.GeneratedProtocolMessageType('Int64List', (_message.Message,), dict(
  DESCRIPTOR = _INT64LIST,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.Int64List)
  ))
_sym_db.RegisterMessage(Int64List)

BytesTuple = _reflection.GeneratedProtocolMessageType('BytesTuple', (_message.Message,), dict(
  DESCRIPTOR = _BYTESTUPLE,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.BytesTuple)
  ))
_sym_db.RegisterMessage(BytesTuple)

Int64Tuple = _reflection.GeneratedProtocolMessageType('Int64Tuple', (_message.Message,), dict(
  DESCRIPTOR = _INT64TUPLE,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.Int64Tuple)
  ))
_sym_db.RegisterMessage(Int64Tuple)

FloatTuple = _reflection.GeneratedProtocolMessageType('FloatTuple', (_message.Message,), dict(
  DESCRIPTOR = _FLOATTUPLE,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.FloatTuple)
  ))
_sym_db.RegisterMessage(FloatTuple)

BytesTupleList = _reflection.GeneratedProtocolMessageType('BytesTupleList', (_message.Message,), dict(
  DESCRIPTOR = _BYTESTUPLELIST,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.BytesTupleList)
  ))
_sym_db.RegisterMessage(BytesTupleList)

Int64TupleList = _reflection.GeneratedProtocolMessageType('Int64TupleList', (_message.Message,), dict(
  DESCRIPTOR = _INT64TUPLELIST,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.Int64TupleList)
  ))
_sym_db.RegisterMessage(Int64TupleList)

FloatTupleList = _reflection.GeneratedProtocolMessageType('FloatTupleList', (_message.Message,), dict(
  DESCRIPTOR = _FLOATTUPLELIST,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.FloatTupleList)
  ))
_sym_db.RegisterMessage(FloatTupleList)

Feature = _reflection.GeneratedProtocolMessageType('Feature', (_message.Message,), dict(
  DESCRIPTOR = _FEATURE,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.Feature)
  ))
_sym_db.RegisterMessage(Feature)

Features = _reflection.GeneratedProtocolMessageType('Features', (_message.Message,), dict(

  FeatureEntry = _reflection.GeneratedProtocolMessageType('FeatureEntry', (_message.Message,), dict(
    DESCRIPTOR = _FEATURES_FEATUREENTRY,
    __module__ = 'Signals_pb2'
    # @@protoc_insertion_point(class_scope:Signals.Features.FeatureEntry)
    ))
  ,
  DESCRIPTOR = _FEATURES,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.Features)
  ))
_sym_db.RegisterMessage(Features)
_sym_db.RegisterMessage(Features.FeatureEntry)

Example = _reflection.GeneratedProtocolMessageType('Example', (_message.Message,), dict(
  DESCRIPTOR = _EXAMPLE,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.Example)
  ))
_sym_db.RegisterMessage(Example)

FeatureList = _reflection.GeneratedProtocolMessageType('FeatureList', (_message.Message,), dict(
  DESCRIPTOR = _FEATURELIST,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.FeatureList)
  ))
_sym_db.RegisterMessage(FeatureList)

FeatureLists = _reflection.GeneratedProtocolMessageType('FeatureLists', (_message.Message,), dict(

  FeatureListEntry = _reflection.GeneratedProtocolMessageType('FeatureListEntry', (_message.Message,), dict(
    DESCRIPTOR = _FEATURELISTS_FEATURELISTENTRY,
    __module__ = 'Signals_pb2'
    # @@protoc_insertion_point(class_scope:Signals.FeatureLists.FeatureListEntry)
    ))
  ,
  DESCRIPTOR = _FEATURELISTS,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.FeatureLists)
  ))
_sym_db.RegisterMessage(FeatureLists)
_sym_db.RegisterMessage(FeatureLists.FeatureListEntry)

SequenceExample = _reflection.GeneratedProtocolMessageType('SequenceExample', (_message.Message,), dict(
  DESCRIPTOR = _SEQUENCEEXAMPLE,
  __module__ = 'Signals_pb2'
  # @@protoc_insertion_point(class_scope:Signals.SequenceExample)
  ))
_sym_db.RegisterMessage(SequenceExample)


_FLOATLIST.fields_by_name['value']._options = None
_INT64LIST.fields_by_name['value']._options = None
_FEATURES_FEATUREENTRY._options = None
_FEATURELISTS_FEATURELISTENTRY._options = None
# @@protoc_insertion_point(module_scope)
