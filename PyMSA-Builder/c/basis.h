#ifndef BASIS_H_
#define BASIS_H_

#include <math.h>

#define NX 6
#define NM 32
#define NP 50

void evmono(double x[NX], double m[NM + 1]);
void evpoly(double m[NM + 1], double p[NP + 1]);
void bemsav(double x[NX], double p[NP + 1]); 

#endif /* BASIS_H_ */
