package nlp.assignments;

import java.util.List;

public class Model2WordAlinger extends Model1WordAlinger {
  double[][] Zss;
  static int MAX = 200;

  public Model2WordAlinger(
    List<WordAlignmentTester.SentencePair> trainingSentencePairs, double ratio, double bucket
  ) {
    super(trainingSentencePairs, ratio, bucket);
  }

  @Override
  protected void initialize() {
    super.initialize();
    Zss = new double[MAX][MAX];
  }

  private double dpZss(int I, int J) {
    if (Zss[I][J] == 0) {
      Zss[I][J] = calcuateSum(I, J);
    }
    if (Zss[I][J] == 0) {
      throw new IllegalStateException();
    }
    return Zss[I][J];
  }


  private double calcuateSum(int I, int J) {
    double sum = 0;
    for (int i = 0; i < I; i++) {
      for (int j = 0; j < J; j++) {
        sum += distanceProb(i, j, I, J);
      }
    }
    return sum;
  }

  @Override
  protected double getP_f_e(String e, String f, int i, int j, int I, int J) {
    if (e.equals("NULL")) return bucket * e2fWords.getCount(e, f);
    if (I >= MAX) {
      I = MAX - 1;
    }
    if (J >= MAX) {
      J = MAX - 1;
    }
    double indexProb = distanceProb(i, j, I, J) / dpZss(I, J);
    double result = (1 - bucket) * e2fWords.getCount(e, f) * indexProb;
    if (Double.isNaN(result)) {
      throw  new IllegalStateException();
    }
    return result;
  }

  private double distanceProb(int i, int j, double I, double J) {
    return Math.exp(-Math.abs(i / I - j / J));
  }
}
