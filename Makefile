default: py

.PHONY: py clean

py:
	cd py && $(MAKE)

clean:
	cd py && $(MAKE) clean
