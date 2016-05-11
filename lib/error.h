#ifndef ERROR_H
#define ERROR_H

class Error {};

class RuntimeError: Error {};

class AssertionError: RuntimeError{};


#endif
