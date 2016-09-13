package nlp.assignments;

import kylm.model.ngram.NgramLM;
import kylm.model.ngram.reader.ArpaNgramReader;
import nlp.langmodel.LanguageModel;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KylmWrapper implements LanguageModel {

  private final NgramLM model;

  public KylmWrapper(NgramLM model) {
    this.model=model;
   // model.setClosed(false);
  }

  public static LanguageModel getModel(String smoothing) throws IOException {
    ArpaNgramReader anr = new ArpaNgramReader();
    NgramLM model = anr.read("model-" + smoothing + ".arpa");
    return new KylmWrapper(model);
  }

  @Override
  public double getSentenceProbability(List<String> sentence) {
    List<String> list = new ArrayList<>();
    for (String str:sentence) {
      list.add(str.substring(0, 1).toUpperCase() + str.substring(1));
    }
    try {
      double pow = Math.pow(10, model.getSentenceProb(sentence.toArray(new String[0])));
      if (pow == 0) return Math.pow(Math.E, -1000);
      return pow;
    } catch (Exception e) {
      return Math.pow(Math.E, -1000);
    }
  }

  @Override
  public List<String> generateSentence() {
    return null;
  }


}
