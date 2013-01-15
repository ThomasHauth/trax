#include "DetectorGeometry.h"
#include <assert.h>

int DetectorGeometry::resolveDetId(uint detId) const {
		int imin = 0; int imax = size();
		// continually narrow search until just one element remains
		while (imin < imax)
		{
			int imid = imin + (imax - imin) / 2;

			// code must guarantee the interval is reduced at each iteration
			assert(imid < imax);
			// note: 0 <= imin < imax implies imid will always be less than imax

			// reduce the search
			if (DetUnit(*this, imid).detId() < detId)
				imin = imid + 1;
			else
				imax = imid;
		}
		// At exit of while:
		//   if A[] is empty, then imax < imin
		//   otherwise imax == imin

		// deferred test for equality
		if ((imax == imin) && (DetUnit(*this, imin).detId() == detId))
			return imin;
		else
			return -1;
	}

