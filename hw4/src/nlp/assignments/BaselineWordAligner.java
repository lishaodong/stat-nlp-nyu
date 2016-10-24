package nlp.assignments;

import static nlp.assignments.WordAlignmentTester.*;

/**
 * Simple alignment baseline which maps french positions to english positions.
 * If the french sentence is longer, all final word map to null.
 */
public class BaselineWordAligner implements WordAligner {
  public Alignment alignSentencePair(SentencePair sentencePair) {
    Alignment alignment = new Alignment();
    int numFrenchWords = sentencePair.getFrenchWords().size();
    int numEnglishWords = sentencePair.getEnglishWords().size();
    for (int frenchPosition = 0; frenchPosition < numFrenchWords; frenchPosition++) {
      int englishPosition = frenchPosition;
      if (englishPosition >= numEnglishWords)
        englishPosition = -1;
      alignment.addAlignment(englishPosition, frenchPosition, true);
    }
    return alignment;
  }
}
