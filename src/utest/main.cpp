#include "utest.h"
int main(int argc, char** argv) {
    UnitTest::getInstance().executeAll();
    return 0;
}
