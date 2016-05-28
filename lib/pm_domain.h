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
  void push(const Index i, const Float x[]) {
    // Push the particle position to packet, send if the packet become full
    vbuf.push_back(x[0]);
    vbuf.push_back(x[1]);
    vbuf.push_back(x[2]);
    vbuf_index.push_back(i);

    if(vbuf.size() >= packet_size)
      send_packet();
  }
  Float xbuf_min, xbuf_max;
  int rank;
  static const int packet_size= 1024/3*3; // number of positions (3*sizeof(Float))
 private:
  std::vector<Float> vbuf;
  std::vector<Index> vbuf_index;
};

struct Packet {
  int dest_rank, n, offset, offset_index;
};

void domain_init(FFT const * const fft, Particles const * const particles);
void domain_send_positions(Particles* const particles);
Pos const * domain_buffer_positions();
Float3* domain_buffer_forces();
int domain_buffer_np();

#endif
