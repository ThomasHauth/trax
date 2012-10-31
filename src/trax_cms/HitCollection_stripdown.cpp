

#include <iostream>


#include <datastructures/HitCollection.h>

int main(){
	HitCollection ht;

	float fX = 23.0f;
	float fY = 5.0f;

	GlobalX gx;
	GlobalY gy;

	ht.setValue(gy, fY);
	ht.setValue(gx, fX);

	const float outx =  ht.getValue( gx );
	const float outy =  ht.getValue( gy );

	std::cout << outx << outy;

	return 0;
}
