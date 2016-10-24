package nlp.assignments;

import nlp.classify.FeatureExtractor;
import nlp.util.Counter;

public class UNKExtractor implements FeatureExtractor<String, String> {

  /**
   * This method takes the list of characters representing the proper name
   * description, and produces a list of features which represent that
   * description. The basic implementation is that only character-unigram
   * features are extracted. An easy extension would be to also throw
   * character bigrams into the feature list, but better features are also
   * possible.
   */
  public Counter<String> extractFeatures(String name) {
    Counter<String> features = new Counter<String>();
    // add character unigram features
    uni(name, features);
    end(name, features);
    begin(name, features);
    //length(name, features);
    //special(name, features);
    //charLength(name, features);

    return features;
  }
  private void uni(String name, Counter<String> features) {
    char[] characters = name.toCharArray();
    for (int i = 0; i < characters.length; i++) {
      char character = characters[i];
      features.incrementCount("UNI-" + character, 1.0);
    }
  }

  private void  end(String name, Counter<String> features) {
    String[] arr = name.split(" ");
    for (int i = 0; i < arr.length; i++) {
      for (int j = 1; j <= 6; j++) {
        if (arr[i].length() < j) {
          continue;
        }
        features.incrementCount("END-" + arr[i].substring(arr[i].length() - j), 1.0);
      }
    }
  }

  private void special(String name, Counter<String> features) {
    char[] characters = name.toCharArray();
    for (int i = 0; i < characters.length; i++) {
      char character = characters[i];
      if (Character.isLetter(character)) {
        continue;
      }
      features.incrementCount("SPECIAL-" + character, 1.0);
    }
  }

  private void  begin(String name, Counter<String> features) {
    String[] arr = name.split(" ");
    for (int i = 0; i < arr.length; i++) {
      for (int j = 1; j <= 6; j++) {
        if (arr[i].length() < j) {
          continue;
        }
        features.incrementCount("BEGIN-" + arr[i].substring(0, j), 1.0);
      }
    }
  }

  private void length(String name, Counter<String> features) {
    String[] arr = name.split(" ");
    features.incrementCount("WORD_COUNT-" + arr.length, 1);
  }

  private void charLength(String name, Counter<String> features) {
    features.incrementCount("CHAR_COUNT-" + name.length(), 1);
  }
}