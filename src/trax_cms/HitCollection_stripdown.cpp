

#include <iostream>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>

int main(){
	HitCollection ht( 400000 );
	clever::context contx;

	HitCollectionTransfer clTrans;

	clTrans.initBuffers( contx, ht );

	clTrans.toDevice( contx, ht );
	clTrans.fromDevice( contx, ht );

	return 0;
}
