#ifndef TIMELOG
void timer(const char name[]) {}
void timer_write_h5(const char filename[]) {}
#else

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <sys/time.h>
#include <hdf5.h>

#include "msg.h"
#include "comm.h"
#include "error.h"
#include "timer.h"

using namespace std;


namespace {
  vector<double> vtime;
  vector<string> vname;
  double t0= 0.0;
}

void timer_write_txt(const char filename[]);
void timer_write_h5(const char filename[]);

static double now()
{
  struct timeval tp;
  gettimeofday(&tp, 0);

  return (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6;
}


void timer(const char name[])
{
  double t= now() - t0;
  if(t0 == 0.0) {
    t0= t;
    t= 0.0;
  }

  vtime.push_back(t);
  vname.push_back(name);
}

void timer_write(const char filebase[])
{
  char filename[128];
  sprintf(filename, "%s.h5", filebase);
  timer_write_h5(filename);

  sprintf(filename, "%s.txt", filebase);
  if(comm_this_node() == 0)
    timer_write_txt(filename);
}

void timer_write_h5(const char filename[])
{
  timer("end");
  
  // Open file
  //H5Eset_auto2(H5E_DEFAULT, NULL, 0);
    
  // Parallel file access
  hid_t plist_file= H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_file, MPI_COMM_WORLD, MPI_INFO_NULL);

  //hid_t file= H5Fopen(filename, H5F_ACC_RDWR, plist_file);
  //if(file < 0) {
  hid_t file= H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_file);
  if(file < 0) {
    msg_printf(msg_error, "Error: unable to create HDF5 file, %s\n", filename);
    throw IOError();
  }
  //}

  // Create group
  /*
  // !!! ToDo needs to parallelise !!!
  hid_t group;
  char name_group[64], name_data[64];
  int ret = sscanf(path, "%s/%s", name_group, name_data);
  if(ret == 2) {
    group= H5Gopen(file, name_group, H5P_DEFAULT);
    if(group < 0) {
      group= H5Gcreate(file, name_group,
		       H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);
      if(group < 0) {
	msg_printf(msg_error, "Error: unable to create group, %s\n",
		   name_group);
	throw IOError();
      }
    }
  }
  else {    
    strncpy(name_data, path, 63);
    group= file;
  }
  
  assert(group >= 0);
  herr_t status_group= H5Gclose(group);
  assert(status_group >= 0);
  */
  //
  const hsize_t ncol= comm_n_nodes();
  const hsize_t nrow= vtime.size();

  // Data structure in memory (local)
  const hsize_t data_size_mem= nrow;
  hid_t memspace= H5Screate_simple(1, &data_size_mem, 0);

  const hsize_t offset_mem= 0;
  const hsize_t stride= 1;
  const hsize_t block_size_mem= 1;
  const hsize_t block_count_mem= nrow;
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET,
		      &offset_mem, &stride, &block_count_mem, &block_size_mem);

  // Data structure in file
  const hsize_t dim= 2;
  const hsize_t data_size_file[]= {nrow, ncol};
  hid_t filespace= H5Screate_simple(dim, data_size_file, NULL);

  // local subset of data for this node
  const hsize_t offset_file[]= {0, hsize_t(comm_this_node())};
  const hsize_t count_file[]= {nrow, 1};

  H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
		      offset_file, NULL, count_file, NULL);

  hid_t dataset= H5Dcreate(file, "time", H5T_IEEE_F64LE, filespace,
			   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert(dataset >= 0);

  hid_t plist_data= H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_data, H5FD_MPIO_COLLECTIVE);
    
  const herr_t status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memspace,
				 filespace, plist_data, &vtime.front());

  H5Pclose(plist_data);
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(dataset);
  
  assert(status >= 0);

  H5Pclose(plist_file);
  H5Fclose(file);
}

void timer_write_txt(const char filename[])
{
  FILE* fp= fopen(filename, "w");

  for(vector<string>::iterator p= vname.begin(); p != vname.end(); ++p)
    fprintf(fp, "%s\n", p->c_str());
  
  fclose(fp);
}

#endif
