#pragma once
#include <cstring>
#include <complex>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <cuComplex.h>
typedef std::string String;
typedef float Real;
typedef std::complex<Real> Complex;

static const cuComplex CUCOMPLEX_ONE = make_cuComplex(1.f,0.f);
static const cuComplex CUCOMPLEX_ZERO = make_cuComplex(0.f,0.f);
