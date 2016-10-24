package nlp.assignments;

import nlp.util.Counter;
import nlp.util.CounterMap;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static nlp.assignments.WordAlignmentTester.*;
public class Model1WordAlinger implements WordAligner {
  protected double bucket;
  double thresholdRatio;
  int numIter = 0;
  boolean verbose = true;
  List<SentencePair> trainingSentencePairs;
  Set<String> englishVacabulary = new HashSet<>();
  Set<String> frenchVacabulary = new HashSet<>();

  CounterMap<String, String> e2fWords = new CounterMap<>();

  public Model1WordAlinger(
    List<SentencePair> trainingSentencePairs, double ratio, double bucket
  ) {
    this.bucket = bucket;
    this.thresholdRatio = ratio;
    this.trainingSentencePairs = new ArrayList<>();
    for (SentencePair pair: trainingSentencePairs) {
      ArrayList<String> englishWordsWithNull =
        new ArrayList<>(pair.getEnglishWords());
      englishWordsWithNull.add("NULL");
      SentencePair newPair = new SentencePair(-1, "", englishWordsWithNull, pair.getFrenchWords());
      this.trainingSentencePairs.add(newPair);
    }
    initialize();
    System.out.println("start EM");
    EM();
    //System.out.println(e2fWords);
  }

  protected void initialize() {
    for (SentencePair pair : trainingSentencePairs) {
      List<String> frenchWords = pair.getFrenchWords();
      List<String> englishWords = pair.getEnglishWords();
      for (String englishWord: englishWords) {
        for (String frenchWord: frenchWords) {
          e2fWords.setCount(englishWord, frenchWord, 1);
        }
      }
    }
    e2fWords.normalize();
  }

  private void EM() {
    double threshold = e2fWords.totalSize() * thresholdRatio;
    double difference = threshold + 1;
    boolean converged = false;
    while (!converged) {
      difference = 0;
      if (verbose && numIter%10==0) {
        System.out.println("iter:"+numIter);
      }
      numIter ++;
      converged = true;
      CounterMap<String, String> e2fCounter = new CounterMap<>();
      Counter<String> totalE = new Counter<>();
      Counter<String> s_totalF = new Counter<>();

      for (WordAlignmentTester.SentencePair pair : trainingSentencePairs) {
        List<String> frenchWords = pair.getFrenchWords();
        List<String> englishWords = pair.getEnglishWords();
        int E = englishWords.size();
        int F = frenchWords.size();
        // compute normalization
        for (int fj = 0; fj < F; fj++) {
          String f = frenchWords.get(fj);
          double s_total_f = 0;
          for (int ei = 0; ei < E; ei++) {
            String e = englishWords.get(ei);
            s_total_f += getP_f_e(e, f, ei, fj, E-1, F);
          }
          if (Double.isNaN(s_total_f)) {
            throw  new IllegalStateException();
          }
          s_totalF.setCount(f, s_total_f);
        }
        // collect counts
        for (int fj = 0; fj < F; fj++) {
          String f = frenchWords.get(fj);
          double s_total_f = s_totalF.getCount(f);
          for (int ei = 0; ei < E; ei++) {
            String e = englishWords.get(ei);
            double increaseAmount = getP_f_e(e, f, ei, fj, E-1, F) / s_total_f;
            if (Double.isNaN(increaseAmount)) {
              throw  new IllegalStateException();
            }
            e2fCounter.incrementCount(e, f, increaseAmount);
            totalE.incrementCount(e, increaseAmount);
          }
        }
      }
      // estimate probabilities
      for (String e : e2fWords.keySet()) {
        Counter<String> fCounter = e2fCounter.getCounter(e);
        for (String f : fCounter.keySet()) {
          double newProb = e2fCounter.getCount(e, f) / totalE.getCount(e);
          double gap = Math.abs(newProb - e2fWords.getCount(e, f));
          difference+=gap;
          if (gap > thresholdRatio) {
            converged = false;
          }
          if (Double.isNaN(newProb)) {
            throw  new IllegalStateException();
          }
          e2fWords.setCount(e, f, newProb);
        }
      }
    }
  }

  @Override
  public Alignment alignSentencePair(SentencePair sentencePair) {
    WordAlignmentTester.Alignment alignment = new WordAlignmentTester.Alignment();
    List<String> frenchWords = sentencePair.getFrenchWords();
    List<String> englishWords = new ArrayList<>(sentencePair.getEnglishWords());
    int F = frenchWords.size();
    englishWords.add("NULL");
    int numFrenchWords = sentencePair.getFrenchWords().size();

    for (int fj = 0; fj < numFrenchWords; fj++) {
      String frenchWord = frenchWords.get(fj);
      String bestEnglishWord= getMostPossibleEnglishWord(frenchWord,
        englishWords, fj, F);
      int englishPosition = bestEnglishWord.equals("NULL")
        ? -1
        : englishWords.indexOf(bestEnglishWord);
      alignment.addAlignment(englishPosition, fj, true);
    }
    return alignment;
  }

  private String getMostPossibleEnglishWord(
    String frenchWord, List<String>englishWords, int fj, int F) {
    int E = englishWords.size() -1;

    String resultWord = null;
    double maxProb = -1;
    for (int ei = 0; ei < englishWords.size(); ei++) {
      String englishWord = englishWords.get(ei);
      double prob = getP_f_e(englishWord, frenchWord, ei, fj, E, F);

      if (prob >= maxProb) {
        maxProb = prob;
        resultWord = englishWord;
      }
    }
    if (resultWord==null) {
      System.out.println();
    }
    return resultWord;
  }

  protected double positionProb(String e, int I) {
    return e.equals("NULL") ? 0.2 : 0.8 / I;
  }

  protected double getP_f_e(String e, String f, int i, int j, int I, int J) {
    return positionProb(e, I) * e2fWords.getCount(e, f);
  }
}
