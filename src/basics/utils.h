#pragma once
#include "xmux.h"

inline bool isPower2(int n) {
    return n > 0 && (n & (n-1)) == 0;
}

void fftshift(XMux<ComplexMatrix>& A);
void ifftshift(XMux<ComplexMatrix>& A);

