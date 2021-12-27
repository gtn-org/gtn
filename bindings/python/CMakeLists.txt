function (add_pybind11_extension ext_name)
  pybind11_add_module(
    ${ext_name}
    ${PROJECT_SOURCE_DIR}/bindings/python/gtn/${ext_name}.cpp
  )
  target_link_libraries(
    ${ext_name}
    PRIVATE
    gtn
  )
  add_dependencies(${ext_name} pybind11)
endfunction()

add_pybind11_extension(_graph) 
add_pybind11_extension(_device)
add_pybind11_extension(_functions)
add_pybind11_extension(_utils)
add_pybind11_extension(_creations)
add_pybind11_extension(_criterions)
add_pybind11_extension(_autograd)
add_pybind11_extension(cuda)
add_pybind11_extension(_parallel)
add_pybind11_extension(_rand)
add_pybind11_extension(_experimental)
