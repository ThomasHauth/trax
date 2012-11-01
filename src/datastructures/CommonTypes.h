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

struct HitId: public clever::UIntItem
{
};
