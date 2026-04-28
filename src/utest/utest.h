#pragma once
#include "types.h"

typedef bool (*UTestFunc)();

class UnitTest {
 private:
  UnitTest() { ; }
    std::map<String,UTestFunc> m_registeredFuncs;

 public:
  static UnitTest& getInstance() {
    static UnitTest _ins;
    return _ins;
  }

  void addTest(const String& key, UTestFunc fp) {
    if(m_registeredFuncs.count(key) > 0) {
      std::cerr << "Duplicated test found: " << key << std::endl;
      exit(1);
    }
    m_registeredFuncs[key] = fp;
  }

  int execute(const String& key, UTestFunc fp);

  int executeAll();

};

struct AddUnitTest {
  AddUnitTest(const String& key, UTestFunc fp) {
    UnitTest::getInstance().addTest(key,fp);
  }
};
