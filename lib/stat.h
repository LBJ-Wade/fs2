#ifndef STAT_H
#define STAT_H 1

void stat_set_filename(const char filename[]);
void stat_write_int(const char group_name[], const char data_name[],
		    int const * const dat, const int n);


#endif
