/**
 * run_tagger reads the trained model from model_file,
 * performs POS tagging on the sents.test based on the model,
 * then outputs the tagged sentences in sents.out
 *
 * @author Huang Lie Jun (A0123994W)
 * @version 1.0
 * @since 2017-10-08
 */
import java.util.*;

public class run_tagger {

    public static void main(String[] args) {
        FileHandler testFile, modelFile, outputFile = null;
        List<String[]> testSents = null;
        Model testModel = null;
        if (args.length >= 3) {
            testFile = new FileHandler(args[0]);
            testFile.readFile();
            testSents = testFile.getFileAsCorpus();
            modelFile = new FileHandler(args[1]);
            testModel = modelFile.readFileAsModel();
            outputFile = new FileHandler(args[2]);
        } else {
            System.err.println("Incorrect number of parameters.");
            System.exit(-1);
        }
        if (testModel != null && outputFile != null) {
            List<List<String>> taggedResult = testModel.tag(testSents);
            List<String> taggedSentences = new ArrayList<String>();
            for (int i = 0; i < taggedResult.size(); i++) {
                String sentence = String.join(" ", taggedResult.get(i));
                taggedSentences.add(sentence);
            }
            String taggedCorpus = String.join("\n", taggedSentences);
            outputFile.writeFile(taggedCorpus);
        }
    }
}

