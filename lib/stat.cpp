#include <string>
#include <cassert>
#include <hdf5.h>
#include "comm.h"
#include "msg.h"
#include "error.h"
#include "stat.h"

using namespace std;

static string stat_filename("stat.h5");

void stat_set_filename(const char filename[])
{
  stat_filename = string(filename);
}

void stat_write_int(const char group_name[],
		    const char data_name[],
		    int const * const dat, const int n)
{
  //if(!stat) return;
  
  //H5Eset_auto2(H5E_DEFAULT, NULL, 0);
  
  // Open File
  hid_t plist_file= H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_file, MPI_COMM_WORLD, MPI_INFO_NULL);

  hid_t file= H5Fopen(stat_filename.c_str(), H5F_ACC_RDWR, plist_file);
  if(file < 0)
    file= H5Fcreate(stat_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_file);

  if(file < 0) {
    msg_printf(msg_error, "Error: unable to open HDF5 file, %s\n",
	       stat_filename.c_str());
    throw IOError();
  }

  // Open Group
  hid_t group= H5Gopen(file, group_name, H5P_DEFAULT);
  if(group < 0) {
    group= H5Gcreate(file, group_name,
		     H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);
    if(group < 0) {
      msg_printf(msg_error, "Error: unable to create a group, %s\n",
		 group_name);
      throw IOError();
    }
  }

  // Setup data structure in memory and in file
  const hsize_t ncol= comm_n_nodes();
  const hsize_t nrow= n;

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

  hid_t dataset= H5Dcreate(group, data_name, H5T_STD_I32LE, filespace,
			   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert(dataset >= 0);

  hid_t plist_data= H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_data, H5FD_MPIO_COLLECTIVE);
    
  const herr_t status = H5Dwrite(dataset, H5T_NATIVE_INT, memspace,
				 filespace, plist_data, dat);

  H5Pclose(plist_data);  
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(dataset);
  
  assert(status >= 0);
  
  H5Gclose(group);
  H5Pclose(plist_file);
  H5Fclose(file);

  msg_printf(msg_verbose, "Wrote %s/%s/%s\n",
	     stat_filename.c_str(), group_name, data_name);
}

