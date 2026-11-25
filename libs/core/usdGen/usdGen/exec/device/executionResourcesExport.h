// Export control for the independently shipped resource-admission DSO.
#ifndef USDGEN_EXECUTION_RESOURCES_EXPORT_H
#define USDGEN_EXECUTION_RESOURCES_EXPORT_H

#if defined(_WIN32)
#  if defined(usdGenExecutionResources_EXPORTS)
#    define USDGEN_EXECUTION_RESOURCES_API __declspec(dllexport)
#  else
#    define USDGEN_EXECUTION_RESOURCES_API __declspec(dllimport)
#  endif
#else
#  define USDGEN_EXECUTION_RESOURCES_API __attribute__((visibility("default")))
#endif

#endif  // USDGEN_EXECUTION_RESOURCES_EXPORT_H
