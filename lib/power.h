#ifndef POWER_H
#define POWER_H 1

#include <gsl/gsl_spline.h>

class PowerSpectrum {
 public:
  PowerSpectrum(const char filename[]);
  ~PowerSpectrum();
  double P(const double k);

 private:
  int n_;
  double* log_k_;
  double* log_P_;
  gsl_interp *interp_;
  gsl_interp_accel *acc_;

  void read_file_(const char filename[]);    
};

class ErrorPowerFile {

};

#endif
