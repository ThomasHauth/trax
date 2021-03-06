#pragma once

#include <clever/clever.hpp>

struct GlobalX: public clever::FloatItem
{
};
struct GlobalY: public clever::FloatItem
{
};

struct GlobalZ: public clever::FloatItem
{
};

struct DetectorId: public clever::UIntItem
{
};

struct DetectorLayer: public clever::UIntItem
{
};

struct HitId: public clever::UIntItem
{
};

struct EventNumber: public clever::UIntItem
{
};

struct NHits: public clever::UIntItem
{
};
struct Offset: public clever::UIntItem
{
};
