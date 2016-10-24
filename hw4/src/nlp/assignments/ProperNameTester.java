package nlp.assignments;

import nlp.classify.FeatureExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * This is the main harness for assignment 2. To run this harness, use
 * <p>
 * java nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p>
 * First verify that the data can be read on your system using the baseline
 * model. Second, find the point in the main method (near the bottom) where a
 * MostFrequentLabelClassifier is constructed. You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them
 * there.
 */
public class ProperNameTester {

  public static class ProperNameFeatureExtractor implements
          FeatureExtractor<String, String> {

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
      word(name, features);
      //uni(name, features);
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
    private void word(String name, Counter<String> features) {
      String[] arr = name.split(" ");
      features.incrementCount("WORD_COUNT-" + arr.length, 1);
      for (int i = 0; i < arr.length; i++) {
        features.incrementCount("WORD-"+arr[i].toLowerCase(), 1.0);
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
        if (Character.isLetterOrDigit(character)) {
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
          features.incrementCount("END-" + arr[i].substring(0, j), 1.0);
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

  private static List<LabeledInstance<String, String>> loadData(
          String fileName) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(fileName));
    List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
    while (reader.ready()) {
      String line = reader.readLine();
      String[] parts = line.split("\t");
      String label = parts[0];
      String name = parts[1];
      LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
              label, name);
      labeledInstances.add(labeledInstance);
    }
    reader.close();
    return labeledInstances;
  }

  private static void testClassifier(
          ProbabilisticClassifier<String, String> classifier,
          List<LabeledInstance<String, String>> testData, boolean verbose) {
    int[][] errors = new int[5][5];

    double numCorrect = 0.0;
    double numTotal = 0.0;
    for (LabeledInstance<String, String> testDatum : testData) {
      String name = testDatum.getInput();
      String label = classifier.getLabel(name);
      double confidence = classifier.getProbabilities(name).getCount(
              label);
      if (label.equals(testDatum.getLabel())) {
        numCorrect += 1.0;
      } else {
        if (verbose) {
          // display an error
          System.err.println("Example: " + name + " guess=" + label
                  + " gold=" + testDatum.getLabel() + " confidence="
                  + confidence);
          //errors[getIndex(testDatum.getLabel())][getIndex(label)] ++;
        }
      }
      numTotal += 1.0;
    }
    double accuracy = numCorrect / numTotal;
    System.out.print(String.format(" & %.1f", accuracy*100));
    //System.out.println(Arrays.deepToString(errors));
  }
  private static int getIndex(String label) {
    if (label.equals("drug")) {
      return 0;
    }
    if (label .equals( "movie")) {
      return 1;
    }
    if (label .equals( "person")) {
      return 2;
    }
    if (label .equals( "place") ){
      return 3;
    }
    if (label .equals( "company")) {
      return 4;
    }
    return -1;
  }




  public static void main(String[] args) throws IOException {
    // Parse command line flags and arguments
    Map<String, String> argMap = CommandLineUtils
            .simpleCommandLineParser(args);

    // Set up default parameters and settings
    String basePath = ".";
    String model = "baseline";
    boolean verbose = false;
    boolean useValidation = true;

    // Update defaults using command line specifications

    // The path to the assignment data
    if (argMap.containsKey("-path")) {
      basePath = argMap.get("-path");
    }
    System.out.println("Using base path: " + basePath);

    // A string descriptor of the model to use
    if (argMap.containsKey("-model")) {
      model = argMap.get("-model");
    }
    System.out.println("Using model: " + model);

    // A string descriptor of the model to use
    if (argMap.containsKey("-test")) {
      String testString = argMap.get("-test");
      if (testString.equalsIgnoreCase("test"))
        useValidation = false;
    }
    System.out.println("Testing on: "
            + (useValidation ? "validation" : "test"));

    // Whether or not to print the individual speech errors.
    if (argMap.containsKey("-verbose")) {
      verbose = true;
    }

    // Load training, validation, and test data
    List<LabeledInstance<String, String>> trainingData = loadData(basePath
            + "/pnp-train.txt");
    List<LabeledInstance<String, String>> validationData = loadData(basePath
            + "/pnp-validate.txt");
    List<LabeledInstance<String, String>> testData = loadData(basePath
            + "/pnp-test.txt");

    // Learn a classifier
    ProbabilisticClassifier<String, String> classifier = null;
    if (model.equalsIgnoreCase("baseline")) {
      classifier = new MostFrequentLabelClassifier.Factory<String, String>()
              .trainClassifier(trainingData);
    } else if (model.equalsIgnoreCase("n-gram")) {
      // TODO: construct your n-gram model here
    } else if (model.equalsIgnoreCase("maxent")) {
      // TODO: construct your maxent model here
//      double[] sigmas = new double[]{0.1, 0.5, 1.0, 2.0, 5.0, 10};
//      int[] iters = new int[]{20, 50, 100, 500};
//      for (double sigma: sigmas) {
//        for(int iter:iters) {
//          ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
//                  sigma, iter, new ProperNameFeatureExtractor());
//          classifier = factory.trainClassifier(trainingData);
//          testClassifier(classifier, (useValidation ? validationData : testData),
//                  verbose);
//
//        }
//        System.out.println();
//      }
          ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
                  2.0, 50, new ProperNameFeatureExtractor());
          classifier = factory.trainClassifier(trainingData);
    } else {
      throw new RuntimeException("Unknown model descriptor: " + model);
    }

    // Test classifier
    testClassifier(classifier, (useValidation ? validationData : testData),
            verbose);
  }


}

