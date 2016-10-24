package nlp.assignments;

import static nlp.assignments.WordAlignmentTester.*;
import nlp.util.Counter;
import nlp.util.CounterMap;

import java.util.List;

public class HeuristicWordAligner implements WordAligner{
  CounterMap<String, String> map = new CounterMap<>();
  Counter<String> bigram = new Counter<>();
  Counter<String> french = new Counter<>();
  Counter<String> english = new Counter<>();

  public HeuristicWordAligner(List<WordAlignmentTester.SentencePair> trainingSentencePairs) {
    for (WordAlignmentTester.SentencePair pair : trainingSentencePairs) {
      List<String> frenchWords = pair.getFrenchWords();
      List<String> englishWords = pair.getEnglishWords();

      for (String englishWord : englishWords) {
        for (String frenchWord : frenchWords) {
          bigram.incrementCount(englishWord + " " + frenchWord, 1);
        }
      }

      for (String englishWord : englishWords) {
        english.incrementCount(englishWord, 1);
      }
      for (String frenchWord : frenchWords) {
        french.incrementCount(frenchWord, 1);
      }
    }
  }
  public WordAlignmentTester.Alignment alignSentencePair(WordAlignmentTester.SentencePair sentencePair) {
    WordAlignmentTester.Alignment alignment = new WordAlignmentTester.Alignment();
    List<String> frenchWords = sentencePair.getFrenchWords();
    List<String> englishWords = sentencePair.getEnglishWords();
    int numFrenchWords = sentencePair.getFrenchWords().size();
    int numEnglishWords = sentencePair.getEnglishWords().size();
    for (int frenchPosition = 0; frenchPosition < numFrenchWords;
         frenchPosition++) {
      String frenchWord = frenchWords.get(frenchPosition);
      String bestEnglishWord= getMostPossibleEnglishWord(frenchWord,
        englishWords);
      int englishPosition = englishWords.indexOf(bestEnglishWord);
      alignment.addAlignment(englishPosition, frenchPosition, true);
    }
    return alignment;
  }

  private String getMostPossibleEnglishWord(String frenchWord, List<String>
    englishWords) {
    Counter<String> counter = map.getCounter(frenchWord);
    String resultWord = null;
    double maxRatio = 0;
    double frenchCount = french.getCount(frenchWord);
    for (String englishWord : englishWords) {
      double ratio = bigram.getCount(englishWord + " " + frenchWord) / (english.getCount(englishWord) * frenchCount);
      if (ratio > maxRatio) {
        maxRatio = ratio;
        resultWord = englishWord;
      }
    }
    return resultWord;
  }
}
