TESTS :=
TESTS += test_pm_density.py test_pm_cic test_fft 


# $(basename names...)

check: $(TESTS)

%: %.py
	for n in 1 2 3 4; do mpirun -n $$n python3 $< || exit 1; done


