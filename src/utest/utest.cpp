#include "utest.h"

#include <iostream>
#include <fstream>
using namespace std;

int UnitTest::execute(const String& key, UTestFunc fp) {
  cout << "running utest: " << key << " ..." << endl;
  auto pass = fp();
  if (pass) {
    cout << key << " runs through successfully" << endl;
    return 0;
  }

  cout << key << " fails" << endl;
  return 1;
}

int UnitTest::executeAll() {
  std::ofstream ofs;
  int fail_count = 0;
  for (const auto& t : m_registeredFuncs) {
    if (execute(t.first, t.second)) {
      if (!ofs.is_open()) ofs.open("utest_fail.list");
      ofs << t.first << endl;
      fail_count++;
    }
  }

  if (ofs.is_open()) ofs.close();

  cout << "Total test: " << m_registeredFuncs.size() << endl;
  cout << "Failure cout: " << fail_count << endl;

  return fail_count;
}
