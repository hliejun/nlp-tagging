/**
 * build_tagger trains the POS tagger with sents.train
 * and tunes the POS tagger parameters with sents.devt,
 * then writes the output statistics to model_file
 *
 * @author Huang Lie Jun (A0123994W)
 * @version 1.0
 * @since 2017-10-08
 */
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

enum Technique {LAPLACE, WITTENBELL, KNESERNEY}
enum Type {WORD, TAG, BOTH}

public class build_tagger {
    /**
     * This function will train a HMM model using training sentences,
     * select the optimal smoothing scheme for the trained model using
     * development sentences, perform a 10-fold validation on training
     * sentences to acquire averaged accuracy and write model results
     * to file for testing and actual tagging.
     *
     * @param trainFile File path to training sentences
     * @param devFile File path to development sentences
     * @param modelFile File path to write model data (params. and prob.)
     */
    public static void main(String[] args) {
        List<String[]> trainCorpus = null;
        List<String[]> devCorpus = null;
        if (args.length >= 3) {
            FileHandler trainFile = new FileHandler(args[0]);
            trainFile.readFile();
            trainCorpus = trainFile.getFileAsCorpus();

            FileHandler devFile = new FileHandler((args[1]));
            devFile.readFile();
            devCorpus = devFile.getFileAsCorpus();

            FileHandler modelFile = new FileHandler(args[2]);
        } else {
            System.err.println("Incorrect number of parameters.");
            System.exit(-1);
        }
        Model posModel = new Model();
        posModel.train(trainCorpus);
        posModel.tune(devCorpus);
        double validatedAccuracy = posModel.crossValidate(trainCorpus, 10);
        System.out.println("Cross-validation accuracy of trained model: " + (validatedAccuracy * 100) + "%");
        Technique bestTechnique = posModel.getBestTechnique();
        HashMap<String, Double> modelTransitionMatrix = posModel.getTransitionProbMatrix();
        HashMap<String, Double> modelEmissionMatrix = posModel.getEmissionProbMatrix();
        HashMap<String, Integer> wordFreqMatrix = posModel.getWordFreq();
        HashMap<String, Integer> tagFreqMatrix = posModel.getTagFreq();
        HashMap<String, Integer> wordTagFreqMatrix = posModel.getWordTagFreq();
        HashMap<String, Integer> prevCurrTagFreqMatrix = posModel.getPrevCurrTagFreq();
        // TODO: Write model attributes to modelFile
    }
}

class FileHandler {
    Path filePath;
    byte[] existingFileData;

    public FileHandler(String pathString) {
        filePath = Paths.get(pathString);
    }

    public void readFile() {
        if (Files.exists(filePath)) {
            try {
                existingFileData = Files.readAllBytes(filePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.err.println("File to be read does not exist.");
        }
    }

    public String getFileAsString() {
        if (existingFileData != null) {
            return new String(existingFileData);
        } else {
            return null;
        }
    }

    public String[] getFileAsSentences() {
        if (existingFileData != null) {
            String linesString = new String(existingFileData);
            return linesString.split("[\\r?\\n]+");
        } else {
            return null;
        }
    }

    public List<String[]> getFileAsCorpus() {
        if (existingFileData != null) {
            String linesString = new String(existingFileData);
            String[] sentences = linesString.split("[\\r?\\n]+");
            List<String[]> corpus = new ArrayList<String[]>();
            for (String sentence : sentences) {
                corpus.add(sentence.split(" "));
            }
            return corpus;
        } else {
            return null;
        }
    }

    public void writeFile(String writeData) {
        try {
            byte[] newFileData = writeData.getBytes("utf-8");
            Files.write(filePath, newFileData, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

class Model {
    private Technique smoothingMode = Technique.LAPLACE;
    private HashMap<String, Integer> wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq;
    private HashMap<String, Double> transitionProbMatrix, emissionProbMatrix;
    private List<String> uniqueWords, uniqueTags;
    private String startTag = "<s>";
    private String separator = "/";

    public Model() {
        super();
    }

    public void train(List<String[]> trainingCorpus) {
        indexCorpus(trainingCorpus);
        buildTransitionMatrix();
        buildEmissionMatrix();
    }

    public double test(List<String[]> testCorpus, Technique smoothingScheme) {
        Set<String> testWords, seenWords, unseenWords;
        HashMap<String, Integer> testWordsFreq;
        int correct = 0, total = 0;
        SmoothScheme smoother = null;
        List<String[]> untaggedTestCorpus = getStrippedCorpus(testCorpus);
        switch (smoothingScheme) {
            case LAPLACE:
                smoother = new Laplace(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, 1);
                break;
            case WITTENBELL:
                testWordsFreq = new HashMap<String, Integer>();
                for (String[] sentence : untaggedTestCorpus) {
                    for (String word : sentence) {
                        incrementFreqTable(testWordsFreq, word);
                    }
                }
                seenWords = wordFreq.keySet();
                unseenWords = new HashSet<String>(testWordsFreq.keySet());
                unseenWords.removeAll(seenWords);
                smoother = new WittenBell(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, seenWords.size(), unseenWords.size());
                break;
            default:
                smoother = new Laplace(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, 1);
                break;
        }
        for (int sentenceIndex = 0; sentenceIndex < untaggedTestCorpus.size(); sentenceIndex++) {
            String[] sentence = untaggedTestCorpus.get(sentenceIndex);
            String[] taggedSentence = testCorpus.get(sentenceIndex);
            double[][] pathProbMatrix = new double[uniqueTags.size() + 1][sentence.length];
            int[][] backpointerMatrix = new int[uniqueTags.size() + 1][sentence.length];
            double maxPathValue = -1;
            int bestPrevTagIndex = -1;
            for (int wordIndex = 0; wordIndex < sentence.length; wordIndex++) {
                String currentWord = sentence[wordIndex];
                for (int tagIndex = 0; tagIndex < uniqueTags.size(); tagIndex++) {
                    Double alpha, beta;
                    String currentTag = uniqueTags.get(tagIndex);
                    if (currentTag == startTag) {
                        continue;
                    } else if (wordIndex == 0) {
                        alpha = transitionProbMatrix.get(startTag + separator + currentTag);
                        alpha = (alpha != null) ? alpha : smoother.getBigramTransition(startTag, currentTag);
                        beta = emissionProbMatrix.get(currentWord + separator + currentTag);
                        beta = (beta != null) ? beta : smoother.getBigramEmission(currentWord, currentTag);
                        pathProbMatrix[tagIndex][wordIndex] = alpha * beta;
                        backpointerMatrix[tagIndex][wordIndex] = -1;
                    } else {
                        bestPrevTagIndex = 0;
                        maxPathValue = 0.0d;
                        for (int prevTagIndex = 0; prevTagIndex < uniqueTags.size(); prevTagIndex++) {
                            String prevTag = uniqueTags.get(prevTagIndex);
                            alpha = transitionProbMatrix.get(prevTag + separator + currentTag);
                            alpha = (alpha != null) ? alpha : smoother.getBigramTransition(prevTag, currentTag);
                            double value = pathProbMatrix[prevTagIndex][wordIndex - 1] * alpha;
                            if (value >= maxPathValue) {
                                maxPathValue = value;
                                bestPrevTagIndex = prevTagIndex;
                            }
                        }
                        beta = emissionProbMatrix.get(currentWord + separator + currentTag);
                        beta = (beta != null) ? beta : smoother.getBigramEmission(currentWord, currentTag);
                        pathProbMatrix[tagIndex][wordIndex] = maxPathValue * beta;
                        backpointerMatrix[tagIndex][wordIndex] = bestPrevTagIndex;
                    }
                }
            }
            int bestEndIndex = 0;
            maxPathValue = 0.0d;
            for (int tagIndex = 0; tagIndex < uniqueTags.size(); tagIndex++) {
                String tag = uniqueTags.get(tagIndex);
                if (tag == startTag) {
                    continue;
                }
                double pathValue = pathProbMatrix[tagIndex][sentence.length - 1];
                if (pathValue >= maxPathValue) {
                    maxPathValue = pathValue;
                    bestEndIndex = tagIndex;
                }
            }
            pathProbMatrix[uniqueTags.size()][sentence.length - 1] = maxPathValue;
            backpointerMatrix[uniqueTags.size()][sentence.length - 1] = bestEndIndex;
            int prevStateIndex = bestEndIndex;
            int prevSequenceIndex = sentence.length - 1;
            List<String> prediction = new ArrayList<String>();
            while (prevStateIndex != -1 && prevSequenceIndex >= 0) {
                String tag = uniqueTags.get(prevStateIndex);
                String word = sentence[prevSequenceIndex];
                prediction.add(0, word + separator + tag);
                prevStateIndex = backpointerMatrix[prevStateIndex][prevSequenceIndex];
                prevSequenceIndex -= 1;
            }
            for (int predictionIndex = 0; predictionIndex < prediction.size(); predictionIndex++) {
                if (prediction.get(predictionIndex).equals(taggedSentence[predictionIndex])) {
                    correct += 1;
                }
                total += 1;
            }
        }
        return (double)correct / (double)total;
    }

    public void tune(List<String[]> testCorpus) {
        Technique[] techniques = new Technique[]{Technique.LAPLACE, Technique.LAPLACE};
        double currentAccuracy = 0, bestAccuracy = 0;
        for (Technique technique : techniques) {
            currentAccuracy = this.test(testCorpus, technique);
            if (currentAccuracy >= bestAccuracy) {
                bestAccuracy = currentAccuracy;
                this.smoothingMode = technique;
            }
        }
        System.out.println("Best Technique: " + this.smoothingMode);
        System.out.println("Best Accuracy: " + bestAccuracy * 100 + "%");
    }

    public double crossValidate(List<String[]> corpus, int n) {
        double averageAccuracy = 0;
        if (n <= 0) {
            System.err.println("Cross validation fold must be positive.");
            return 0;
        } else {
            List<String[]> validationCorpus, trainingCorpus, remCorpus;
            int intervalSize = (int)Math.ceil((double)corpus.size() / n);
            Model cvModel = new Model();
            for (int start = 0; start < corpus.size(); start += intervalSize) {
                int end = start + intervalSize;
                end = (end <= corpus.size()) ? end : corpus.size();
                validationCorpus = new ArrayList<String[]>(corpus.subList(start, end));
                trainingCorpus = (start == 0)
                        ? new ArrayList<String[]>(corpus.subList(end, corpus.size()))
                        : new ArrayList<String[]>(corpus.subList(0, start));
                if (end < corpus.size()) {
                    remCorpus = new ArrayList<String[]>(corpus.subList(end, corpus.size()));
                    trainingCorpus.addAll(remCorpus);
                }
                cvModel.train(trainingCorpus);
                double accuracy = cvModel.test(validationCorpus, this.getBestTechnique());
                averageAccuracy += accuracy;
            }
            return averageAccuracy / n;
        }
    }

    public Technique getBestTechnique() {
        return this.smoothingMode;
    }

    public HashMap<String, Double> getTransitionProbMatrix() {
        return this.transitionProbMatrix;
    }

    public HashMap<String, Double> getEmissionProbMatrix() {
        return this.emissionProbMatrix;
    }

    public HashMap<String, Integer> getWordFreq() {
        return this.wordFreq;
    }

    public HashMap<String, Integer> getTagFreq() {
        return this.tagFreq;
    }

    public HashMap<String, Integer> getWordTagFreq() {
        return this.wordTagFreq;
    }

    public HashMap<String, Integer> getPrevCurrTagFreq() {
        return this.prevCurrTagFreq;
    }

    private interface Smoothing {
        public double getBigramTransition(String prevTag, String currTag);
        public double getBigramEmission(String word, String tag);
        public double getUnknownWordBigramEmission(String wordTag, String prevWordTag);
    }

    private abstract class SmoothScheme implements Smoothing {
        private HashMap<String, Integer> wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq;

        public SmoothScheme(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq) {
            super();
            this.wordFreq = wordFreq;
            this.tagFreq = tagFreq;
            this.wordTagFreq = wordTagFreq;
            this.prevCurrTagFreq = prevCurrTagFreq;
        }
    }

    private class Laplace extends SmoothScheme {
        int laplaceFactor;

        public Laplace(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
            this.laplaceFactor = 1;
        }

        public Laplace(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq, int laplaceFactor) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
            this.laplaceFactor = laplaceFactor;
        }

        public double getBigramTransition(String prevTag, String currTag) {
            return ((double)countPrevCurrTag(prevTag, currTag) + 1) / ((double)countTag(prevTag) + (double)tagFreq.size());
        }

        public double getBigramEmission(String word, String tag) {
            return ((double)countWordTag(word, tag) + 1) / ((double)countTag(tag) + (double)tagFreq.size());
        }

        public double getUnknownWordBigramEmission(String wordTag, String prevWordTag) {
            // (KIV) TODO: P(word|tag) = P(word|tag) * P(capital-word|tag) * P(ending-hyph|tag)
            return 0;
        }
    }

    private class WittenBell extends SmoothScheme {
        int seen, unseen;

        public WittenBell(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq, int seen, int unseen) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
            this.seen = seen;
            this.unseen = unseen;
        }

        public double getBigramTransition(String prevTag, String currTag) {
            //    (Seen tag)   : P(tag|prev-tag) = Count(prev-tag,tag) / [Count(prev-tag) + seen]
            // TODO: -> (Unseen tag) : P(tag|prev-tag) = seen / {unseen * [Count(prev-tag) + seen]}
            return 0;
        }

        public double getBigramEmission(String word, String tag) {
            // TODO: -> (Unseen word) : P(word|tag) = seen / {unseen * [Count(tag) + seen]}
            return 0;
        }

        public double getUnknownWordBigramEmission(String wordTag, String prevWordTag) {
            // (Unseen tag) : P(word|tag) = seenTypes / {unseenTags * [Count(tag) + seenTags]}
            // (KIV) TODO: (Unseen word) : P(unknown-word|tag) = P(unknown-word|tag) * P(capital-word|tag) * P(ending-hyph|tag)
            return 0;
        }
    }

    private class KneserNey extends SmoothScheme {

        public KneserNey(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
        }

        public double getBigramTransition(String prevTag, String currTag) {
            // TODO: Transition Probability
            return 0;
        }

        public double getBigramEmission(String word, String tag) {
            // TODO: Emission Probability
            return 0;
        }

        public double getUnknownWordBigramEmission(String wordTag, String prevWordTag) {
            // TODO: Unknown Word Probability
            return 0;
        }
    }

    private void indexCorpus(List<String[]> corpus) {
        String prevWord, currWord, prevTag, currTag, prevCurrTag;
        String[] currWordTag;
        String prev = "", curr = "";
        wordFreq = new HashMap<String, Integer>();
        tagFreq = new HashMap<String, Integer>();
        wordTagFreq = new HashMap<String, Integer>();
        prevCurrTagFreq = new HashMap<String, Integer>();
        for (String[] sentence : corpus) {
            for (int index = 0; index < sentence.length; index++) {
                curr = sentence[index];
                currWordTag = splitElement(curr);
                currWord = currWordTag[0];
                currTag = currWordTag[1];
                incrementFreqTable(wordFreq, currWord);
                incrementFreqTable(tagFreq, currTag);
                incrementFreqTable(wordTagFreq, curr);
                if (index == 0) {
                    incrementFreqTable(tagFreq, startTag);
                    prevTag = startTag;
                } else {
                    prev = sentence[index - 1];
                    prevTag = splitElement(prev)[1];
                }
                prevCurrTag = prevTag + separator + currTag;
                incrementFreqTable(prevCurrTagFreq, prevCurrTag);
            }
        }
        uniqueWords = new ArrayList<String>(wordFreq.keySet());
        Collections.sort(uniqueWords);
        uniqueTags = new ArrayList<String>(tagFreq.keySet());
        Collections.sort(uniqueTags);
    }

    private void buildTransitionMatrix() {
        String prevTag, currTag, prevCurrTag;
        transitionProbMatrix = new HashMap<String, Double>();
        for (int row = 0; row < uniqueTags.size(); row++) {
            currTag = uniqueTags.get(row);
            for (int col = 0; col < uniqueTags.size(); col++) {
                prevTag = uniqueTags.get(col);
                prevCurrTag = prevTag + separator + currTag;
                transitionProbMatrix.put(prevCurrTag, (double)countPrevCurrTag(prevTag, currTag) / (double)countTag(prevTag));
            }
        }
    }

    private void buildEmissionMatrix() {
        String currWord, currTag, wordTag;
        emissionProbMatrix = new HashMap<String, Double>();
        for (int row = 0; row < uniqueWords.size(); row++) {
            currWord = uniqueWords.get(row);
            for (int col = 0; col < uniqueTags.size(); col++) {
                currTag = uniqueTags.get(col);
                wordTag = currWord + separator + currTag;
                emissionProbMatrix.put(wordTag, (double)countWordTag(currWord, currTag) / (double)countTag(currTag));
            }
        }
    }

    private List<String[]> getStrippedCorpus(List<String[]> taggedCorpus) {
        List<String[]> untaggedTestSent = new ArrayList<String[]>();
        for (String[] sentence : taggedCorpus) {
            String[] strippedSentence = new String[sentence.length];
            for (int index = 0; index < sentence.length; index++) {
                strippedSentence[index] = splitElement(sentence[index])[0];
            }
            untaggedTestSent.add(strippedSentence);
        }
        return untaggedTestSent;
    }

    private String[] splitElement(String element) {
        int index = element.lastIndexOf(separator);
        String word = element.substring(0, index);
        String tag = element.substring(index, element.length()).replace(separator, "");
        String[] splitString = new String[]{word, tag};
        return splitString;
    }

    private int countWord(String word) {
        Integer wordCount = wordFreq.get(word);
        return (wordCount != null) ? (int)wordCount : 0;
    }

    private int countTag(String tag) {
        Integer tagCount = tagFreq.get(tag);
        return (tagCount != null) ? (int)tagCount : 0;
    }

    private int countWordTag(String word, String tag) {
        String wordTag = word + separator + tag;
        Integer wordTagCount = wordTagFreq.get(wordTag);
        return (wordTagCount != null) ? (int)wordTagCount : 0;
    }

    private int countPrevCurrTag(String prevTag, String currTag) {
        String prevCurrTag = prevTag + separator + currTag;
        Integer prevCurrTagCount = prevCurrTagFreq.get(prevCurrTag);
        return (prevCurrTagCount != null) ? (int)prevCurrTagCount : 0;
    }

    private void incrementFreqTable(HashMap<String, Integer> table, String key) {
        Integer value = table.get(key);
        value = (value != null) ? value + 1 : 1;
        table.put(key, value);
    }
}