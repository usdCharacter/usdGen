// usdGen — engine core export macro.
//
// libusdGen.so is built with hidden visibility (CXX_VISIBILITY_PRESET hidden)
// and LINKER:--exclude-libs,ALL so that gate B-1 keeps third-party symbols
// out of its dynamic table. Symbols that are part of the installed M0 public
// surface must therefore be marked USDGEN_CORE_API explicitly; at M0 that is
// usdGen::GetVersionString() only.
#ifndef USDGEN_EXPORT_H
#define USDGEN_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGen_EXPORTS)
#    define USDGEN_CORE_API __declspec(dllexport)
#  else
#    define USDGEN_CORE_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_CORE_API __attribute__((visibility("default")))
#endif

#endif  // USDGEN_EXPORT_H
