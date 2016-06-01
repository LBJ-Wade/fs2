TESTS :=
TESTS += test_pm_force
TESTS += test_particles_h5 test_pm_density test_pm_cic test_fft 


# $(basename names...)

check: $(TESTS)

%: %.py
	for n in 1 2 3 4; do mpirun -n $$n python3 $< || exit 1; done

force_double.h5: create_force_h5.py
	python3 $<

test_pm_force: test_pm_force.py force_double.h5

.PHONY: pep8

pep8:
	pep8 *.py

