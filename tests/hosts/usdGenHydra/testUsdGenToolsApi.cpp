// T1 — the C ABI the usdview expression editor calls
// (libs/hosts/usdGenHydra/usdGenImaging/usdGenToolsApi.h).
//
// Covers what the editor depends on: a clean compile reports OK, a broken one
// reports a diagnostic with a usable line and column, the domain is honoured,
// and both list entry points obey the size-then-read protocol and produce
// parseable records.

#include "usdGenImaging/usdGenToolsApi.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int gFailures = 0;

void Check(bool condition, const char *what)
{
    if (condition) {
        std::printf("ok: %s\n", what);
    } else {
        std::printf("FAIL: %s\n", what);
        ++gFailures;
    }
}

std::vector<std::string> Split(std::string const &text, char delimiter)
{
    std::vector<std::string> parts;
    std::string current;
    for (char ch : text) {
        if (ch == delimiter) { parts.push_back(current); current.clear(); }
        else current.push_back(ch);
    }
    parts.push_back(current);
    return parts;
}

// Read one of the two list entry points through the documented protocol.
template <typename Call>
std::string ReadList(Call call, const char *what)
{
    const int needed = call(nullptr, 0);
    Check(needed > 1, (std::string(what) + " reports a size when asked for one").c_str());
    if (needed <= 1) return "";
    // A buffer one byte short must write nothing and still report the size.
    std::vector<char> tooSmall(size_t(needed) - 1, '\xAB');
    const int again = call(tooSmall.data(), needed - 1);
    Check(again == needed,
          (std::string(what) + " reports the same size for a short buffer").c_str());
    Check(tooSmall[0] == '\xAB',
          (std::string(what) + " writes nothing into a short buffer").c_str());

    std::vector<char> buffer(size_t(needed), '\0');
    const int written = call(buffer.data(), needed);
    Check(written == needed, (std::string(what) + " fills an exact buffer").c_str());
    Check(std::strlen(buffer.data()) == size_t(needed) - 1,
          (std::string(what) + " NUL-terminates").c_str());
    return std::string(buffer.data());
}

}  // namespace

int main()
{
    Check(UsdGenTools_ApiVersion() == USDGEN_TOOLS_API_VERSION,
          "the ABI reports the version this test was built against");

    char errors[4096];

    // --- compiling -------------------------------------------------------
    int status = UsdGenTools_CompileExpression(
        "$value * (0.15 + 0.85 * (1 - $t) * (1 - $t))",
        USDGEN_TOOLS_DOMAIN_POINT, 1, errors, sizeof(errors));
    Check(status == USDGEN_TOOLS_OK, "a valid point-domain expression compiles");
    Check(errors[0] == '\0', "a clean compile leaves the error buffer empty");

    // $t is defined per curve and per CV, never once per groom, so the same
    // text must be refused in the groom domain. This is the check that proves
    // the editor's domain combo actually reaches the engine.
    status = UsdGenTools_CompileExpression(
        "$value * $t", USDGEN_TOOLS_DOMAIN_GROOM, 1, errors, sizeof(errors));
    Check(status == USDGEN_TOOLS_DIAGNOSTICS,
          "a variable outside its domain is refused");

    status = UsdGenTools_CompileExpression(
        "$value * (1 - ", USDGEN_TOOLS_DOMAIN_POINT, 1, errors, sizeof(errors));
    Check(status == USDGEN_TOOLS_DIAGNOSTICS, "an unbalanced expression is refused");
    {
        const std::vector<std::string> fields = Split(Split(errors, '\n')[0], '\t');
        Check(fields.size() >= 3, "a diagnostic has line, column and message");
        if (fields.size() >= 3) {
            Check(std::stoi(fields[0]) >= 1, "the diagnostic carries a line number");
            Check(std::stoi(fields[1]) >= 1, "the diagnostic carries a column");
            Check(!fields[2].empty(), "the diagnostic carries a message");
            Check(fields[2].find('\n') == std::string::npos,
                  "the message is one line, so a status strip can show it");
        }
    }

    // An unsupported function is reported against its own position in the
    // source, which is what the editor underlines.
    status = UsdGenTools_CompileExpression(
        "$value\n  * frobnicate($t)", USDGEN_TOOLS_DOMAIN_POINT, 1, errors,
        sizeof(errors));
    Check(status == USDGEN_TOOLS_DIAGNOSTICS, "an unknown function is refused");
    {
        const std::vector<std::string> fields = Split(Split(errors, '\n')[0], '\t');
        if (fields.size() >= 3) {
            Check(std::stoi(fields[0]) == 2,
                  "the character offset maps onto the right line");
            Check(std::stoi(fields[1]) > 1,
                  "the character offset maps onto a column within that line");
        }
    }

    Check(UsdGenTools_CompileExpression(nullptr, USDGEN_TOOLS_DOMAIN_POINT, 1,
                                        errors, sizeof(errors)) ==
              USDGEN_TOOLS_BAD_ARGS,
          "a NULL source is a bad-argument error, not a crash");
    Check(UsdGenTools_CompileExpression("$value", 99, 1, errors, sizeof(errors)) ==
              USDGEN_TOOLS_BAD_ARGS,
          "an unknown domain is a bad-argument error");
    Check(UsdGenTools_CompileExpression("$value", USDGEN_TOOLS_DOMAIN_POINT, 9,
                                        errors, sizeof(errors)) ==
              USDGEN_TOOLS_BAD_ARGS,
          "an impossible component count is a bad-argument error");
    Check(UsdGenTools_CompileExpression("$value", USDGEN_TOOLS_DOMAIN_POINT, 1,
                                        nullptr, 0) == USDGEN_TOOLS_OK,
          "a status-only compile needs no error buffer");

    // --- the variable list ------------------------------------------------
    const std::string variables = ReadList(
        [](char *buf, int len) {
            return UsdGenTools_ListVariables(USDGEN_TOOLS_DOMAIN_GROOM, buf, len);
        },
        "ListVariables");
    bool sawValue = false, sawT = false, tIsInvalidForGroom = false;
    bool everyRecordComplete = true;
    for (std::string const &line : Split(variables, '\n')) {
        const std::vector<std::string> fields = Split(line, '\t');
        if (fields.size() < 6) { everyRecordComplete = false; continue; }
        if (fields[0] == "$value") sawValue = true;
        if (fields[0] == "$t") {
            sawT = true;
            tIsInvalidForGroom = fields[4] == "0";
            Check(fields[3].find("point") != std::string::npos,
                  "$t lists the domains it is defined in");
        }
        if (fields[0] == "$P") {
            Check(fields[1] == "float32" && fields[2] == "3",
                  "$P reports its scalar type and component count");
        }
    }
    Check(everyRecordComplete, "every variable record has all six fields");
    Check(sawValue && sawT, "the variable list carries $value and $t");
    Check(tIsInvalidForGroom,
          "$t is marked invalid for the groom domain rather than hidden");

    // --- the function list ------------------------------------------------
    const std::string functions = ReadList(
        [](char *buf, int len) { return UsdGenTools_ListFunctions(buf, len); },
        "ListFunctions");
    bool sawClamp = false, sawTernary = false;
    bool everyFunctionCategorised = true;
    std::string clampCategory, ternaryCategory, sinCategory;
    for (std::string const &line : Split(functions, '\n')) {
        const std::vector<std::string> fields = Split(line, '\t');
        if (fields.size() < 4) continue;
        // The browser groups by category, so a record without one would leave
        // a function in an unnamed bin.
        if (fields.size() < 5 || fields[4].empty()) everyFunctionCategorised = false;
        if (fields[0] == "clamp") {
            sawClamp = true;
            Check(fields[1] == "clamp(x, lo, hi)", "clamp reports its signature");
            Check(!fields[2].empty(), "clamp carries a doc line");
            Check(fields[3].find('|') != std::string::npos,
                  "an insertion carries the caret marker");
            if (fields.size() >= 5) clampCategory = fields[4];
        }
        if (fields[0] == "sin" && fields.size() >= 5) sinCategory = fields[4];
        if (fields[0] == "?:") {
            sawTernary = true;
            if (fields.size() >= 5) ternaryCategory = fields[4];
        }
        // Everything the list advertises must actually compile, or the
        // browser would offer a user expressions the engine refuses.
        if (fields[0].size() && std::isalpha(static_cast<unsigned char>(fields[0][0]))) {
            // Rebuild the call from the signature with 1 for every argument,
            // rather than substituting names, so a name that also occurs in
            // the function's own spelling cannot corrupt the probe. A
            // trailing "..." (curve()'s repeating knot triples) is not
            // itself an argument, so it is cut before counting commas: the
            // shortest legal call is everything up to the ellipsis.
            const size_t open = fields[1].find('(');
            if (open == std::string::npos) continue;
            std::string inside = fields[1].substr(open + 1);
            const size_t close = inside.rfind(')');
            if (close != std::string::npos) inside.resize(close);
            const size_t ellipsis = inside.find("...");
            if (ellipsis != std::string::npos) inside.resize(ellipsis);
            while (!inside.empty() &&
                   (inside.back() == ' ' || inside.back() == ','))
                inside.pop_back();
            // String parameters (the sampling functions) take a string
            // literal, and an optional one is left out: its default is the
            // only spelling the function is guaranteed to accept.
            std::vector<std::string> parameters;
            if (!inside.empty()) parameters = Split(inside, ',');
            std::string probe = fields[1].substr(0, open) + "(";
            size_t written = 0;
            for (std::string parameter : parameters) {
                while (!parameter.empty() && parameter.front() == ' ')
                    parameter.erase(parameter.begin());
                const bool isString = parameter.rfind("string ", 0) == 0;
                if (isString && parameter.find('=') != std::string::npos) break;
                probe += written++ ? ", " : "";
                probe += isString ? "\"1\"" : "1";
            }
            probe += ")";
            const int probeStatus = UsdGenTools_CompileExpression(
                probe.c_str(), USDGEN_TOOLS_DOMAIN_POINT, 1, errors,
                sizeof(errors));
            if (probeStatus != USDGEN_TOOLS_OK) {
                std::printf("FAIL: listed function '%s' does not compile as "
                            "'%s': %s\n", fields[0].c_str(), probe.c_str(),
                            errors);
                ++gFailures;
            }
        }
    }
    Check(sawClamp, "the function list carries clamp");
    Check(sawTernary, "the function list carries the ternary operator");
    Check(everyFunctionCategorised,
          "every function record carries a non-empty category");
    Check(ternaryCategory == "Operators",
          "the ternary is filed under Operators, not with the functions");
    Check(sinCategory == "Trigonometry",
          "sin is filed under Trigonometry");
    Check(!clampCategory.empty() && clampCategory != "Other",
          "clamp is filed under a named category rather than the leftovers "
          "bin");

    if (gFailures) {
        std::printf("FAIL: %d check(s) failed\n", gFailures);
        return 1;
    }
    std::printf("PASS: usdGenTools C ABI\n");
    return 0;
}
