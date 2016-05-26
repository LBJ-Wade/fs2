#ifndef DOMAIN_H
#define DOMAIN_H 1

#include <vector>
#include "config.h"
#include "particle.h"
#include "fft.h"

class Domain {
 public:
  void clear();
  void send_packet();
  void push(const float_t x[]) {
    // Push the particle position to packet, send if the packet become full
    buf.push_back(x[0]);
    buf.push_back(x[1]);
    buf.push_back(x[2]);

    if(buf.size() >= packet_size)
      send_packet();
  }
  float_t xbuf_min, xbuf_max;
  int rank;
  static const int packet_size= 1024; // number of positions (3*sizeof(float_t))
 private:
  std::vector<float_t> buf;
};

struct Packet {
  int dest_rank, n, offset;
};

void domain_init(FFT const * const fft, Particles const * const particles);
void domain_send_positions(Particles* const particles);
Vec3 const * domain_buffer_positions();
int domain_buffer_np();
#endif
