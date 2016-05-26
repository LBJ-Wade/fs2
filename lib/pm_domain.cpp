#include <iostream>
#include <vector>
#include <deque>
#include <cstdlib>
#include <cassert>
#include <mpi.h>

#include "msg.h"
#include "comm.h"
#include "util.h"
#include "pm_domain.h"
#include "error.h"

using namespace std;

static int nc;
static int nbuf, nbuf_alloc;
static MPI_Win win_nbuf, win_pos;
static float_t x_left, x_right;
static float_t* buf_pos= 0;
static vector<Domain> decomposition;
static deque<Packet>  packets_sent;

static inline void send(const int i, const float_t x[], const float_t boxsize);

static void packets_clear();
static void packets_flush();

void domain_init(FFT const * const fft, Particles const * const particles)
{
  if(buf_pos)
    return;

  // Initialise static variables  
  nc= fft->nc;

  const float_t boxsize= particles->boxsize;
  x_left= boxsize/nc*(fft->local_ix0 + 1);
  x_right= boxsize/nc*(fft->local_ix0 + fft->local_nx - 1);

  // Create MPI Windows
  MPI_Win_create(&nbuf, sizeof(int), sizeof(int), MPI_INFO_NULL,
		 MPI_COMM_WORLD, &win_nbuf);

  nbuf_alloc= particles->np_allocated;
  MPI_Win_allocate(sizeof(float_t)*3*nbuf_alloc,
		   sizeof(float_t),  MPI_INFO_NULL, MPI_COMM_WORLD,
		   &buf_pos, &win_pos);

  size_t size_buf= sizeof(float_t)*3*nbuf_alloc;
  
  if(buf_pos == 0) {
    msg_printf(msg_fatal,
       "Error: unable to allocate %lu MBytes for PM domain buffer\n", size_buf);
    throw MemoryError();
  }

  msg_printf(msg_verbose,
	     "PM domain buffer %d MB allocated\n", mbytes(size_buf));


  // Create the decomposition, a vector of domains.
  const float_t xbuf[2]= {boxsize*(fft->local_ix0 - 1)/nc,
			  boxsize*(fft->local_ix0 + fft->local_nx + 1)/nc};
  const int n= comm_n_nodes();

  /*
  if(n == 1) {
    
    Domain d;
    d.rank= 0;
    d.xbuf_min= xbuf[0];
    d.xbuf_max= xbuf[1];
    printf("domain-1 %.1f %.1f\n", xbuf[0], xbuf[1]);
    decomposition.push_back(d);

    return;
  }
  */
  
  float_t* const xbuf_all= (float_t*) malloc(sizeof(float_t)*2*n);
  assert(xbuf_all);
  
  MPI_Allgather(xbuf, 2, FLOAT_TYPE, xbuf_all, 2, FLOAT_TYPE, MPI_COMM_WORLD);

  const int n_dest= n - 1;
  const int this_node= comm_this_node();

  printf("buf %d: %.2f %.2f %.2f %.2f\n", this_node, xbuf_all[0], xbuf_all[1], xbuf_all[2], xbuf_all[3]);

  
  decomposition.reserve(n_dest);
  Domain d;
  for(int i=1; i<=n/2; ++i) {
    int i_plus= (this_node + i) % n;
    assert(i_plus != this_node);
    d.rank= i_plus;
    d.xbuf_min= xbuf_all[2*i_plus];
    d.xbuf_max= xbuf_all[2*i_plus + 1];
    decomposition.push_back(d);

    int i_minus= (this_node - i + n) % n;
    assert(i_minus != this_node);

    if(i_minus != i_plus) {
      d.rank= i_minus;
      d.xbuf_min= xbuf_all[2*i_minus];
      d.xbuf_max= xbuf_all[2*i_minus + 1];
      //printf("domain-minus %d %d %.2f %.2f\n", this_node, i_minus, d.xbuf_min, d.xbuf_max);

      decomposition.push_back(d);
    }
  }

  assert(decomposition.size() == n_dest);
  
  free(xbuf_all);  
}

void domain_send_positions(Particles* const particles)
{
  // Prerequisit: domain_init()
  assert(buf_pos);

  msg_printf(msg_verbose, "sending positions\n");

  nbuf= 0;
  packets_sent.clear();
  packets_clear();
  
  const int this_node= comm_this_node();
  //const int n_nodes= comm_n_nodes();
  const int np= particles->np_local;
  const float_t boxsize= particles->boxsize;

  //const float_t offset_left= this_node == 0 ? boxsize : 0;
  //const float_t offset_right= this_node == n_nodes ? -boxsize : 0;

  //msg_printf(msg_debug, "x_left= %.2f\n", x_left); throw "debug";
  
  MPI_Win_fence(0, win_pos);

  Particle* const p= particles->p;
  for(int i=0; i<np; ++i) {
    periodic_wrapup_p(p[i], boxsize);
    assert(0.0f <= p[i].x[0] && p[i].x[0] <= boxsize);

    // Node: possibility of local_nx = 1 or 0 exists

    msg_printf(msg_debug, "x= %.2f\n", p[i].x[0]);
    
    if(p[i].x[0] < x_left) {
      //msg_printf(msg_debug, "left %.2f\n", p[i].x[0] + offset_left);
      send(i, p[i].x, boxsize);
    }
    if(p[i].x[0] > x_right) {
      //msg_printf(msg_debug, "right %.2f\n", p[i].x[0] + offset_right);
      send(i, p[i].x, boxsize);
    }
  }

  packets_flush();

  MPI_Win_fence(0, win_pos);

  printf("Remote particles in %d: %d\n", this_node, nbuf);
  for(int i=0; i<nbuf; ++i) {
    printf("%d %.1f %.1f %.1f\n", this_node,
	   buf_pos[3*i], buf_pos[3*i+1], buf_pos[3*i+2]);
    
  }
}

Vec3 const * domain_buffer_positions()
{
  return (Vec3 const *) buf_pos;
}

int domain_buffer_np()
{
  return nbuf;
}
  
void send(const int i, const float_t x[], const float_t boxsize)
{
  // ToDo: this is naive linear search all; many better way possible
  printf("%lu domains\n", decomposition.size());
  for(vector<Domain>::iterator
	dom= decomposition.begin(); dom != decomposition.end(); ++dom) {
    printf("send %.2f %.2f %.2f [%.2f, %.2f]\n",
	   x[0], x[1], x[2], dom->xbuf_min, dom->xbuf_max);
    if((dom->xbuf_min < x[0] && x[0] < dom->xbuf_max) ||
       (dom->xbuf_min < x[0] - boxsize && x[0] - boxsize < dom->xbuf_max) ||
       (dom->xbuf_min < x[0] + boxsize && x[0] + boxsize < dom->xbuf_max))
      dom->push(x);
  }
}

void packets_clear()
{
  for(vector<Domain>::iterator
	dom= decomposition.begin(); dom != decomposition.end(); ++dom)
    dom->clear();
}

void packets_flush()
{
  for(vector<Domain>::iterator
	dom= decomposition.begin(); dom != decomposition.end(); ++dom) {
    dom->send_packet();
  }
}

void Domain::clear()
{
  buf.clear();
}

void Domain::send_packet()
{
  assert(buf.size() % 3 == 0);
  const int nsend= buf.size() / 3;
  
  if(nsend == 0) {
    msg_printf(msg_debug, "No particle copy to node %d\n", rank);
    return;
  }

  // offset= rank::nbuf
  // rank::nbuf += nsend
  int offset;
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_nbuf);
  MPI_Get_accumulate(&nsend, 1, MPI_INT, &offset, 1, MPI_INT,
		     rank, 0, 1, MPI_INT, MPI_SUM, win_nbuf);
  MPI_Win_unlock(rank, win_nbuf);

  if(offset + nsend >= nbuf_alloc) {
    msg_printf(msg_fatal,
	       "Error: pm buffer overflow: %d allocated, need more than %d\n",
	       nbuf_alloc, offset + nsend);
    throw RuntimeError();
  }
  MPI_Put(&buf.front(), nsend*3, FLOAT_TYPE,
	  rank, offset*3, nsend*3, FLOAT_TYPE, win_pos);

  msg_printf(msg_debug, "sending packet %d particles to %d, offset= %d\n",
	     nsend, rank, offset);
  
  // Record the information for force get
  Packet pkt;
  pkt.dest_rank= rank;
  pkt.offset= offset;
  pkt.n= nsend;
  
  packets_sent.push_back(pkt);
  
  buf.clear();
}
