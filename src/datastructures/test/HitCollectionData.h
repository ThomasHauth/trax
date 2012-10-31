#pragma once

#include "../HitCollection.h"

namespace HitCollectionData
{
void generatHitTestData(HitCollection & ht)
{
	const unsigned int hitCount = 20000;
	for (unsigned int i = 0; i < hitCount; i++)
	{
		ht.addWithValue(10.f * float(i), float(i), 1.0f / float(i+1), i % 10);
	}
}
}
