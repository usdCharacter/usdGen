// usdGen imaging library — public header placeholder.
// M0 keeps the imaging layer thin: the four registrations (two scene index
// plugins, five prim adapters, one API-schema adapter) and pass-through
// scene index. UsdGenImagingRegistry / session / publisher land in M1.
#ifndef USDGEN_IMAGING_H
#define USDGEN_IMAGING_H

#include "pxr/pxr.h"

PXR_NAMESPACE_USING_DIRECTIVE

// Note (sol S-8): the M0 imaging surface has no free version function of its
// own. The canonical M0 version API is usdGen::GetVersionString() in
// usdGen/usdGen.h (and usdGenMath::GetVersionString() in
// usdGenMath/usdGenMath.h). The former placeholder declaration
// usdGen::GetImagingVersionString() had no definition anywhere and was
// removed so the installed surface carries no undefined promise; the M1
// registry (UsdGenImagingRegistry) is where imaging identity/version
// surfaces will live.

#endif // USDGEN_IMAGING_H
