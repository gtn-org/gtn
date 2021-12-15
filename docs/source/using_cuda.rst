.. _using_cuda:

Using CUDA
===========

After constructing a graph, it can be copied to the GPU device with the
method :cpp:func:`cuda()`. If multiple devices are available the graph will be
moved to the currently active device (see :cpp:func:`getDevice` and
:cpp:func:`setDevice`).

::

  Graph g;
  g.addNode(true);
  g.addNode(false, true);
  g.addArc(0, 1, 0, 0, 0.5);

  // Copy the graph to the GPU device
  g = g.cuda();

  // Check if the graph is on the GPU
  g.isCuda()

  // Copy the GPU graph back to the host
  g = g.cpu()


Not all graph methods and functions are supported on the GPU. For example
graphs may not be modified on the GPU. Functions with CUDA support will
automatically detect if a graph is on the host or device and dispatch to the
correct back-end. For example ``compose(g1, g2)`` will run on the same device
that ``g1`` and ``g2`` are on (an error is thrown if they are not on the same
device).
