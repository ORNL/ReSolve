#ifndef RESOLVE_VERSION_H
#define RESOLVE_VERSION_H


/**
 * Sets major, minor, and patch versions for current ReSolve build. The user is
 * responsible for free'ing this memory.
 */
int ResolveVersionGetVersion(int *, int *, int *);

/**
 * Sets string with build version for current ExaGO build in format
 * "major.minor.patch". The user is responsible for free'ing this memory.
 */
int ReSolveVersionGetVersionStr(std::string &);


#endif