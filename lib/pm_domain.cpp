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

static inline void send(const int i, const float_t x[], const float_t shift);

static void packets_clear();
static void packets_flush();

void domain_init(FFT const * const fft, Particles const * const particles)
{
  if(buf_pos)
    return;

  // Initialise static variables  
  nc= fft->nc;

  const float_t boxsize= particles->boxsize;
  const float_t eps= boxsize/nc;
  x_left= boxsize/nc*fft->local_ix0 + eps;
  x_right= boxsize/nc*(fft->local_ix0 + fft->local_nx) - eps;

  // Create MPI Windows
  MPI_Win_create(&nbuf, sizeof(int), sizeof(int), MPI_INFO_NULL,
		 MPI_COMM_WORLD, &win_nbuf);

  nbuf_alloc= particles->np_allocated;
  MPI_Win_allocate(sizeof(float_t)*3*nbuf_alloc,
		   sizeof(float_t)*3,  MPI_INFO_NULL, MPI_COMM_WORLD,
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
  float_t* const xbuf_all= (float_t*) malloc(sizeof(float_t)*2*n);
  assert(xbuf_all);
  
  MPI_Allgather(xbuf, 2, FLOAT_TYPE, xbuf_all, 2, FLOAT_TYPE, MPI_COMM_WORLD);

  int n_dest= n == 1 ? 1 : n - 1;

  const int this_node= comm_this_node();
  
  decomposition.reserve(n_dest);
  Domain d;
  for(int i=1; i<=n/2; ++i) {
    int i_plus= (this_node + i) % n;
    assert(i_plus != this_node);
    //cerr << "i_plus " << i_plus << endl;
    d.rank= i_plus;
    d.xbuf_min= xbuf[2*i_plus];
    d.xbuf_min= xbuf[2*i_plus + 1];
    decomposition.push_back(d);

    int i_minus= (this_node - i + n) % n;
    assert(i_minus != this_node);
    //cerr << "i_minus " << i_minus << endl;

    if(i_minus != i_plus) {
      d.rank= i_minus;
      d.xbuf_min= xbuf[2*i_minus];
      d.xbuf_min= xbuf[2*i_minus + 1];
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
  const int n_nodes= comm_n_nodes();
  const int np= particles->np_local;
  const float_t boxsize= particles->boxsize;

  const float_t offset_left= this_node == 0 ? boxsize : 0;
  const float_t offset_right= this_node == n_nodes ? -boxsize : 0;

  //msg_printf(msg_debug, "x_left= %.2f\n", x_left); throw "debug";
  
  MPI_Win_fence(0, win_pos);

  Particle* const p= particles->p;
  for(int i=0; i<np; ++i) {
    periodic_wrapup_p(p[i], boxsize);
    assert(0.0f <= p[i].x[0] && p[i].x[0] <= boxsize);

    // Node: possibility of local_nx = 1 or 0 exists
    
    if(p[i].x[0] < x_left) {
      //msg_printf(msg_debug, "%.2f\n", p[i].x[0] + offset_left);
      send(i, p[i].x, offset_left);
    }
    if(p[i].x[0] > x_right) {
      //msg_printf(msg_debug, "%.2f\n", p[i].x[0] + offset_right);
      send(i, p[i].x, offset_right);
    }
  }

  packets_flush();

  MPI_Win_fence(0, win_pos);
}

Vec3 const * domain_buffer_positions()
{
  return (Vec3 const *) buf_pos;
}

int domain_buffer_np()
{
  return nbuf;
}
  
void send(const int i, const float_t x[], const float_t shift)
{
  // ToDo: this is naive linear search all; many better way possible
  for(vector<Domain>::iterator
	dom= decomposition.begin(); dom != decomposition.end(); ++dom) {
    if(dom->xbuf_min < x[0] && x[0] < dom->xbuf_max) // debug!!!
      dom->push(x, shift);
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

  MPI_Put(&buf.front(), nsend, FLOAT_TYPE,
	  rank, offset, nsend, FLOAT_TYPE, win_pos);

  msg_printf(msg_debug, "sending packet %d -> %d, offset= %d\n",
	     nsend, rank, offset);
  
  // Record the information for force get
  Packet pkt;
  pkt.dest_rank= rank;
  pkt.offset= offset;
  pkt.n= nsend;
  
  packets_sent.push_back(pkt);
  
  buf.clear();
}
