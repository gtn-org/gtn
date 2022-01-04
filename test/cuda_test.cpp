#include "catch.hpp"
#include <thread>

#include <gtn/gtn.h>

using namespace gtn;

TEST_CASE("test cuda utils", "[cuda]") {
    CHECK(cuda::isAvailable());
    int numDevices = cuda::deviceCount();
    CHECK(numDevices > 0);

    int device = cuda::getDevice();
    CHECK(device == 0);

    cuda::setDevice(numDevices - 1);
    device = cuda::getDevice();
    CHECK(device == numDevices - 1);

    // Launch work in main thread stream
    detail::HDSpan<float> a(1 << 10, 1.0, Device::CUDA);
    detail::HDSpan<float> b(1 << 10, 0.0, Device::CUDA);
    detail::HDSpan<float> c(1 << 10, 1000.0, Device::CUDA);
    auto event = cuda::Event();
    for (int i = 0; i < 1000; ++i) {
      add(a, b, b);
    }
    event.record();

    // Equality check in a different stream
    auto eq = std::async([&b, &c, &event](){
        event.wait();
        return b == c;
      });
    CHECK(eq.get());
}

TEST_CASE("test graph cuda", "[cuda]") {
  {
    Graph g;
    CHECK(!g.isCuda());
    // Moving empty graphs works
    CHECK(equal(g.cuda().cpu(), g));
    CHECK(equal(g.to(Device::CUDA).cpu(), g));
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 1, 0.5);
    // cpu to cpu is a no-op
    CHECK(g.id() == g.cpu().id());
    CHECK(g.id() == g.to(Device::CPU).id());
    auto gdev = g.cuda();
    CHECK(gdev.numNodes() == g.numNodes());
    CHECK(gdev.numArcs() == g.numArcs());
    CHECK(gdev.isCuda());
    CHECK(gdev.device() == Device::CUDA);
    CHECK(gdev.item() == 0.5);
    CHECK_THROWS(gdev.arcSort());
    // gpu to gpu on the same device is a no-op
    CHECK(gdev.id() == gdev.cuda().id());
    // Copying to another device or between devices
    if (cuda::deviceCount() > 1) {
      CHECK(equal(g.cuda(Device{Device::CUDA, 1}).cpu(), g));
      CHECK(equal(gdev.cuda(Device{Device::CUDA, 1}).cpu(), g));
    }
    auto ghost = gdev.cpu();
    CHECK(!ghost.isCuda());
    CHECK(equal(ghost, g));
  }

  {
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0);
    auto gdev = g.cuda();
    auto gout = Graph(nullptr, {gdev});
    CHECK(gout.isCuda());
    CHECK(gout.device() == gdev.device());
  }

  {
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0, 1, 0.5);
    g.addArc(0, 1, 1, 2, 0.3);
    g.addArc(1, 2, 0, 1, 0.7);
    g.addArc(1, 2, 1, 0, 1.7);
    auto gdev = g.cuda();
    CHECK(equal(g, gdev.cpu()));
    auto gdevCopy = Graph::deepCopy(gdev);
    CHECK(equal(g, gdevCopy.cpu()));
  }

  {
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0);

    Graph grad;
    grad.addNode(true);
    grad.addNode(false, true);
    grad.addArc(0, 1, 0, 0, 1.0);

    // Check adding gradients works.
    g = g.cuda();
    grad = grad.cuda();
    g.addGrad(grad);
    CHECK(g.grad().cpu().item() == 1.0);
    g.addGrad(grad.weights());
    CHECK(g.grad().cpu().item() == 2.0);

    // Check throws for wrong devices
    CHECK_THROWS(g.addGrad(grad.cpu()));
  }
}

TEST_CASE("test graph data cuda", "[cuda]") {
  Graph g;
  g.addNode();
  g.addNode();
  g.addArc(0, 1, 0);
  g = g.cuda();

  auto check_all = [](auto &data) {
    CHECK(data.startIds.isCuda());
    CHECK(data.acceptIds.isCuda());
    CHECK(data.start.isCuda());
    CHECK(data.accept.isCuda());
    CHECK(data.inArcOffset.isCuda());
    CHECK(data.outArcOffset.isCuda());
    CHECK(data.inArcs.isCuda());
    CHECK(data.outArcs.isCuda());
    CHECK(data.ilabels.isCuda());
    CHECK(data.olabels.isCuda());
    CHECK(data.srcNodes.isCuda());
    CHECK(data.dstNodes.isCuda());
  };
  check_all(g.getData());

  g = Graph(nullptr, {g});
  CHECK(g.isCuda());
  check_all(g.getData());
}
