
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(device, m) {
  py::enum_<DeviceType>(m, "DeviceType")
    .value("CPU", DeviceType::CPU)
    .value("CUDA", DeviceType::CUDA);

  m.attr("CPU") = DeviceType::CPU;
  m.attr("CUDA") = DeviceType::CUDA;
  
  auto&& device_class = py::class_<Device>(m, "Device")
      .def(py::init<DeviceType>(), "type"_a)
      .def(py::init<DeviceType, int>(), "type"_a, "index"_a);

  device_class.def(pybind11::self == pybind11::self);
  device_class.def(pybind11::self != pybind11::self);

}
