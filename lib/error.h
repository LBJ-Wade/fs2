#ifndef ERROR_H
#define ERROR_H

class Error {};

class RuntimeError: Error {};

class AssertionError: RuntimeError{};

class IOError: RuntimeError {};

class MemoryError: RuntimeError {};

#endif
