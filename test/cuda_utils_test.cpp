#include "catch.hpp"

#include "gtn/graph.h"
#include "gtn/utils.h"

using namespace gtn;

TEST_CASE("test cuda graph equality", "[cuda utils]") {
  {
    // Empty graph is equal to itself
    Graph g1;
    Graph g2;
    CHECK(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Different start node
    Graph g1;
    g1.addNode(true);

    Graph g2;
    g2.addNode(false);
    CHECK_FALSE(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Simple equality
    Graph g1;
    g1.addNode(true);
    g1.addNode();

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    CHECK(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Different arc label
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addArc(0, 1, 0);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addArc(0, 1, 1);
    CHECK_FALSE(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Different arc weight
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addArc(0, 1, 0, 0, 1.2);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addArc(0, 1, 0, 0, 2.2);
    CHECK_FALSE(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Self loop in g1
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 0);
    g1.addArc(0, 1, 1);
    g1.addArc(1, 1, 1);
    g1.addArc(1, 2, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0);
    g2.addArc(0, 1, 1);
    g2.addArc(1, 2, 2);
    CHECK_FALSE(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Equals
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 0, 2.1);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(1, 1, 1, 1, 4.1);
    g1.addArc(1, 2, 2, 2, 5.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, 2.1);
    g2.addArc(0, 1, 1, 1, 3.1);
    g2.addArc(1, 1, 1, 1, 4.1);
    g2.addArc(1, 2, 2, 2, 5.1);
    CHECK(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Different arc order
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(0, 1, 0, 0, 2.1);
    g1.addArc(1, 1, 1, 1, 4.1);
    g1.addArc(1, 2, 2, 2, 5.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, 2.1);
    g2.addArc(0, 1, 1, 1, 3.1);
    g2.addArc(1, 2, 2, 2, 5.1);
    g2.addArc(1, 1, 1, 1, 4.1);
    CHECK_FALSE(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Repeat arcs
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(0, 1, 1, 1, 4.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 1, 1, 3.1);
    g2.addArc(0, 1, 1, 1, 4.1);
    g2.addArc(0, 1, 1, 1, 4.1);
    CHECK_FALSE(equal(g1.cuda(), g2.cuda()));
  }

  {
    // Transducer with different outputs
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 1, 2.1);
    g1.addArc(1, 1, 1, 3, 4.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 1, 2.1);
    g2.addArc(1, 1, 1, 4, 4.1);
    CHECK_FALSE(equal(g1.cuda(), g2.cuda()));
  }
}
