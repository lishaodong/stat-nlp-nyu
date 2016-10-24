package nlp.assignments;

import nlp.util.Counter;

import java.util.*;

public class ViterbiDecoder<S> implements POSTaggerTester.TrellisDecoder<S> {
  static class Node<S> {
    Node previousNode;
    double logProb = Double.NEGATIVE_INFINITY;
    S state;

    public Node(S state) {
      this.state = state;
    }
    public Node(S state, double logProb) {
      this.state = state;
      this.logProb = logProb;
    }
  }


  @Override
  public List<S> getBestPath(POSTaggerTester.Trellis<S> trellis) {
    Map<S, Node> states = new HashMap<>();
    Map<S, Node> nextStates = new HashMap<>();
    S currentState = trellis.getStartState();

    states.put(currentState, new Node(currentState, 0));

    while (!currentState.equals(trellis.getEndState())) {
      for (S state: states.keySet()) {
        Node thisNode = states.get(state);
        Counter<S> transitions = trellis.getForwardTransitions(state);
        Set<S> nextStatesFromThisState = transitions.keySet();
        for (S nextState: nextStatesFromThisState) {
          currentState = nextState;
          if (nextStates.containsKey(nextState)) {
            Node thatNode = nextStates.get(nextState);
            double newProb = thisNode.logProb + transitions.getCount(nextState);
            if (newProb > thatNode.logProb) {
              thatNode.logProb = newProb;
              thatNode.previousNode = thisNode;
            }
          } else {
            double newProb = thisNode.logProb + transitions.getCount(nextState);
            Node newNode = new Node(nextState, newProb);
            newNode.previousNode = thisNode;
            nextStates.put(nextState, newNode);
          }
        }
      }
      states = nextStates;
      nextStates = new HashMap<>();
    }
    Node<S> node = states.get(currentState);
    List<S> results = new ArrayList<>();
    while (node.previousNode!=null) {
      results.add(node.state);
      node = node.previousNode;
    }
    results.add(node.state);

     Collections.reverse(results);
    return results;
  }
}
