import unittest
import gtn


class CudaTestCase(unittest.TestCase):

    @unittest.skipIf(not gtn.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
      num_devices = gtn.cuda.device_count()
      self.assertTrue(num_devices > 0)

      device = gtn.cuda.get_device()
      self.assertEqual(device, 0)

      gtn.cuda.set_device(num_devices - 1)
      device = gtn.cuda.get_device()
      self.assertEqual(device, num_devices - 1)

    @unittest.skipIf(not gtn.cuda.is_available(), "CUDA not available")
    def test_graph_cuda(self):
      g = gtn.Graph()
      self.assertFalse(g.is_cuda())

      self.assertTrue(gtn.equal(g.cuda().cpu(), g))
      g.add_node(True)
      g.add_node(False, True)
      g.add_arc(0, 1, 0, 1, 0.5)

      self.assertTrue(gtn.equal(g, g.cpu()))
      gdev = g.cuda()
      self.assertEqual(gdev.num_nodes(), g.num_nodes())
      self.assertEqual(gdev.num_arcs(), g.num_arcs())
      self.assertTrue(gdev.is_cuda())
      self.assertEqual(gdev.item(), 0.5)
      self.assertRaises(ValueError, gdev.arc_sort)

      if gtn.cuda.device_count() > 1:
        gpu1 = gtn.Device(gtn.CUDA, 1) 
        self.assertTrue(gtn.equal(g.cuda(gpu1).cpu(), g))
        self.assertTrue(gtn.equal(gdev.cuda(gpu1).cpu(), g))

      ghost = gdev.cpu()
      self.assertFalse(ghost.is_cuda())
      self.assertTrue(gtn.equal(ghost, g))

    @unittest.skipIf(gtn.cuda.is_available(), "CUDA available")
    def test_graph_nocuda(self):
      self.assertFalse(gtn.cuda.is_available());
      self.assertRaises(ValueError, gtn.cuda.device_count)
      self.assertRaises(ValueError, gtn.cuda.get_device)
      self.assertRaises(ValueError, gtn.cuda.set_device, 0)


if __name__ == "__main__":
    unittest.main()
