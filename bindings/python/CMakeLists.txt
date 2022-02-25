function (add_pybind11_extension ext_name rel_path)
  if ("${rel_path}" STREQUAL "")
    pybind11_add_module(
      ${ext_name}
      ${PROJECT_SOURCE_DIR}/bindings/python/gtn/${ext_name}.cpp
    )
  else()
    pybind11_add_module(
      ${ext_name}
      ${PROJECT_SOURCE_DIR}/bindings/python/gtn/${rel_path}/${ext_name}.cpp
    )  
    set_target_properties(${ext_name} PROPERTIES OUTPUT_NAME ${rel_path}/${ext_name})
  endif()
  target_link_libraries(
    ${ext_name}
    PRIVATE
    gtn
  )
  add_dependencies(${ext_name} pybind11)
endfunction()

add_pybind11_extension(graph "") 
add_pybind11_extension(device "")
add_pybind11_extension(functions "")
add_pybind11_extension(utils "")
add_pybind11_extension(creations "")
add_pybind11_extension(autograd "")
add_pybind11_extension(cuda "")
add_pybind11_extension(parallel "")
add_pybind11_extension(rand "")

add_pybind11_extension(criterion criterion)
