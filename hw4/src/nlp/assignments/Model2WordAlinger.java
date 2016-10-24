package nlp.assignments;

import java.util.List;

public class Model2WordAlinger extends Model1WordAlinger {
  private final double bucket;
  double[][] Zss;

  public Model2WordAlinger(
    List<WordAlignmentTester.SentencePair> trainingSentencePairs, double ratio, double bucket
  ) {
    super(trainingSentencePairs, ratio);
    this.bucket = bucket;
  }

  @Override
  protected void initialize() {
    super.initialize();
    int MAX = 180;
    Zss = new double[MAX][MAX];
    for (int I = 1; I < MAX; I++) {
      for (int J = 1; J < MAX; J++) {
        Zss[I][J] = calcuateSum(I, J);
      }
    }
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
    return (1 -bucket) * e2fWords.getCount(e, f) * distanceProb(i, j, I, J) / Zss[I][J];
  }

  private double distanceProb(int i, int j, double I, double J) {
    return Math.exp(-Math.abs(i / I - j / J));
  }
}
