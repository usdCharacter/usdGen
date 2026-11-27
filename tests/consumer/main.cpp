// usdGen M0 consumer probe: link the installed usdGen and call its one
// canonical M0 API (sol S-8).
#include "usdGen/api/usdGen.h"
#include <cstdio>

int main() {
    const std::string v = usdGen::GetVersionString();
    std::printf("%s\n", v.c_str());
    return v.find("usdGen") == std::string::npos ? 1 : 0;
}
