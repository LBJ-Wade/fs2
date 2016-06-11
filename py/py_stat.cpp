#include "pm_domain.h"
#include "stat.h"
#include "py_stat.h"


PyObject* py_stat_set_filename(PyObject* self, PyObject* args)
{
  char const * filename;
  if(!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  
  stat_set_filename(filename);
  Py_RETURN_NONE;
}

PyObject* py_stat_record_pm_nbuf(PyObject* self, PyObject* args)
{
  // _stat_pm_nbuf(group_name)
  char const * group_name;

  if(!PyArg_ParseTuple(args, "s", &group_name)) {
    return NULL;
  }

  const int dat= pm_domain_nbuf();
  stat_write_int(group_name, "pm_nbuf", &dat, 1);

  Py_RETURN_NONE;
}
