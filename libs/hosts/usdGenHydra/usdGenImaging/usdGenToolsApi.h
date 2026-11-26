/* usdGenImaging/usdGenToolsApi.h — the C ABI the usdview authoring tools call.
 *
 * Separate from cApi.h on purpose. cApi.h is the frozen C4 contract for the
 * interactive grooming session (18 entry points, plan/08-tools.md governs);
 * this header is the small, additive surface the SeExpr expression editor
 * needs to compile and describe an expression without embedding a copy of the
 * language in Python. It carries no session state, touches no stage and no
 * GPU, and is safe to call from any thread at any time.
 *
 * Every entry point is extern "C" and never lets an exception cross the
 * boundary. Strings are UTF-8 and always NUL-terminated on success.
 *
 * Sizing protocol for the two list entry points: they return the number of
 * bytes required INCLUDING the terminating NUL. Call once with (buf=NULL,
 * len=0) to size, allocate, call again. When `len` is large enough the text
 * is written; when it is not, nothing is written and the required size is
 * still returned. A negative return is an error.
 */
#ifndef USDGEN_IMAGING_TOOLS_API_H
#define USDGEN_IMAGING_TOOLS_API_H

#include "usdGenImaging/api.h"

#ifndef USDGENIMAGING_API
#define USDGENIMAGING_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes for UsdGenTools_CompileExpression. */
#define USDGEN_TOOLS_OK          0 /* compiles clean                        */
#define USDGEN_TOOLS_DIAGNOSTICS 1 /* did not compile; errBuf holds records */
#define USDGEN_TOOLS_BAD_ARGS    2 /* NULL source, or a domain/components
                                      value outside the accepted set        */

/* Evaluation domains, matching usdGen::expr::Domain and the spellings of the
 * usdGen:evaluation customData on the consuming attribute. */
#define USDGEN_TOOLS_DOMAIN_GROOM     1
#define USDGEN_TOOLS_DOMAIN_PRIMITIVE 2
#define USDGEN_TOOLS_DOMAIN_POINT     4

/* Bumped whenever the delimited-text record layouts below change, so a
 * Python plugin from a different build refuses to misparse them.
 *
 * 2: UsdGenTools_ListFunctions gained a trailing <category> field. */
#define USDGEN_TOOLS_API_VERSION 2
int USDGENIMAGING_API UsdGenTools_ApiVersion(void);

/* Compile `source` for `domain` (one of the three USDGEN_TOOLS_DOMAIN_*
 * values) producing `components` (1-4) floats, and report what the engine
 * frontend says about it.
 *
 * On USDGEN_TOOLS_DIAGNOSTICS, errBuf receives one record per diagnostic,
 * records separated by '\n' and fields by '\t':
 *
 *     <line>\t<column>\t<message>
 *
 * line and column are 1-based positions into `source`; both are 0 when the
 * diagnostic carries no position. The text is truncated (still NUL-terminated)
 * rather than failing if errBuf is too small; 4096 bytes holds any diagnostic
 * the frontend currently produces. errBuf may be NULL when errLen is 0, which
 * asks for the status only. */
int USDGENIMAGING_API UsdGenTools_CompileExpression(const char *source, int domain,
                                                    int components, char *errBuf,
                                                    int errLen);

/* The expression variable table, one record per line, fields tab-separated:
 *
 *     <name>\t<scalarType>\t<components>\t<domains>\t<valid>\t<doc>
 *
 * name        includes the leading '$'.
 * scalarType  "float32", "float64", "int32", "uint32", ... or "invalid" for
 *             the polymorphic $value, whose type is the consumer's.
 * components  decimal; "0" for $value, again because the consumer decides.
 * domains     comma-separated subset of groom,primitive,point in which the
 *             variable is defined.
 * valid       "1" when the variable is available in `domain`, else "0", so a
 *             browser can grey out the rest instead of hiding them. Pass 0 for
 *             `domain` to get "1" everywhere.
 * doc         one line of prose.
 *
 * Every variable is always listed, whatever `domain` is. */
int USDGENIMAGING_API UsdGenTools_ListVariables(int domain, char *buf, int len);

/* The functions and operators an expression may use, one record per line,
 * fields tab-separated:
 *
 *     <name>\t<signature>\t<doc>\t<insert>\t<category>
 *
 * `insert` is the text an editor should put at the cursor; a '|' in it marks
 * where the cursor should land afterwards, and there is at most one.
 *
 * `category` is the group a browser should file the function under ("Math",
 * "Trigonometry", "Noise", "Curves", ...). It is the engine's own category
 * when usdGen::expr::FunctionInfo carries one and a name-based classification
 * otherwise, so the field is always present and never empty. */
int USDGENIMAGING_API UsdGenTools_ListFunctions(char *buf, int len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* USDGEN_IMAGING_TOOLS_API_H */
