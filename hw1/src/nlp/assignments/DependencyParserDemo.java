package nlp.assignments;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.Tree;

import java.io.StringReader;
import java.util.Arrays;
import java.util.List;

/**
 * Demonstrates how to first use the tagger, then use the NN dependency
 * parser. Note that the parser will not work on untagged text.
 *
 * @author Jon Gauthier
 */
public class DependencyParserDemo {
  public static void main(String[] args) {
    String modelPath = "english_UD.gz";
    String taggerPath = "stanford-postagger-2015-12-09/models/english-left3words-distsim.tagger";

    for (int argIndex = 0; argIndex < args.length; ) {
      switch (args[argIndex]) {
        case "-tagger":
          taggerPath = args[argIndex + 1];
          argIndex += 2;
          break;
        case "-model":
          modelPath = args[argIndex + 1];
          argIndex += 2;
          break;
        default:
          throw new RuntimeException("Unknown argument " + args[argIndex]);
      }
    }

    String text = "I can almost always tell when movies use fake dinosaurs.";

   LexicalizedParser lp = LexicalizedParser.loadModel("englishPCFG.ser.gz");

lp.setOptionFlags(new String[]{"-maxLength", "80", "-retainTmpSubcategories"});

String sent = "I clever can";
Tree parse = (Tree) lp.parse(sent);
double score = parse.score();
    System.out.println(score);
  }
}
