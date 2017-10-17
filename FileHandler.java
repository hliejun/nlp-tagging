import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

public class FileHandler {
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

    public Model readFileAsModel() {
        Model importedModel = null;
        if (Files.exists(filePath)) {
            try {
                FileInputStream fileInput = new FileInputStream(filePath.toString());
                ObjectInputStream objectInput = new ObjectInputStream(fileInput);
                importedModel = (Model) objectInput.readObject();
                objectInput.close();
                fileInput.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            System.err.println("File to be read does not exist.");
        }
        return importedModel;
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

    public void writeFile(Model modelToWrite) {
        try {
            FileOutputStream fileOutput = new FileOutputStream(filePath.toString());
            ObjectOutputStream objectOutput = new ObjectOutputStream(fileOutput);
            objectOutput.writeObject(modelToWrite);
            objectOutput.flush();
            objectOutput.close();
            fileOutput.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}