package me.qinchao.captcha;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.util.*;
import java.util.stream.Collectors;


/**
 * Created by sulvto on 17-10-5.
 */

public class CaptchaFileChooser {
    private static Logger log = LoggerFactory.getLogger(CaptchaFileChooser.class);


    /*
    Create a popup window to allow you to chose an image file to test against the
    trained Neural Network
    Chosen images will be automatically
    scaled to 28*28 grayscale
     */
    public static String fileChose() {
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION) {
            File file = fc.getSelectedFile();
            return file.getAbsolutePath();
        } else {
            return null;
        }
    }

    public static void main(String[] args) throws Exception {
        int height = 30;
        int width = 80;
        int channels = 1;

        // recordReader.getLabels()
        // In this version Labels are always in order
        // So this is no longer needed

        List<String> jpgLabelList = Arrays.asList("___0", "___1", "___2", "___3", "___4", "___5", "___6", "___7", "___8", "___9"
                , "__0_", "__1_", "__2_", "__3_", "__4_", "__5_", "__6_", "__7_", "__8_", "__9_"
                , "_0__", "_1__", "_2__", "_3__", "_4__", "_5__", "_6__", "_7__", "_8__", "_9__"
                , "0___", "1___", "2___", "3___", "4___", "5___", "6___", "7___", "8___", "9___"
        );
        Collections.sort(jpgLabelList);

        //LOAD NEURAL NETWORK

        // Where to save model
        File locationToSave = new File("trained_myjpg_model.zip");
        // Check for presence of saved model
        if (locationToSave.exists()) {
            System.out.println("\n######Saved Model Found######\n");
        } else {
            System.out.println("\n\n#######File not found!#######");
            System.out.println("This example depends on running ");
            System.out.println("MnistImagePipelineExampleSave");
            System.out.println("Run that Example First");
            System.out.println("#############################\n\n");


            System.exit(0);
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("*********TEST YOUR IMAGE AGAINST SAVED NETWORK********");

        // pop up file chooser
        String filechose = fileChose();

        //  pop up file chooser
        // FileChose is a string we will need a file

        File file = new File(filechose);

        // Use NativeImageLoader to convert to numerical matrix

        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Get the image into an INDarray

        INDArray image = loader.asMatrix(file);

        // 0-255
        // 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        // Pass through to neural Net

        INDArray output = model.output(image);

        log.info("## The FILE CHOSEN WAS " + filechose);
        log.info("## The Neural Nets Pediction ##");
        log.info("## list of probabilities per label ##");


        Map<String, Float> labels = labels(output, jpgLabelList);

        log.info(output.toString());
        log.info(jpgLabelList.toString());
        // print
        labels.entrySet().forEach(stringFloatEntry -> {
            log.info("[{} : {}]",stringFloatEntry.getKey(),stringFloatEntry.getValue());
        });
    }

    static Map<String, Float> labels(INDArray indArray, List<String> jpgLabelList) {
        List<Map.Entry<String, Float>> result = new ArrayList<>();
        for (int i = 0; i <= 9; i++) {
            float v = Float.parseFloat(indArray.getColumn(i).toString());
            if (v > 0) {
                String label = jpgLabelList.get(i);
                result.add(new HashMap.SimpleEntry(label, v));
            }
        }

        List<Map.Entry<String, Float>> temp = new ArrayList<>(result);
        result.clear();

        for (int i = 10; i <= 19; i++) {

            float v = Float.parseFloat(indArray.getColumn(i).toString());
            if (v > 0) {
                int finalI = i;
                temp.forEach(stringFloatEntry -> {
                    char[] chars = stringFloatEntry.getKey().toCharArray();
                    chars[1] = jpgLabelList.get(finalI).toCharArray()[1];
                    result.add(new HashMap.SimpleEntry(new String(chars), v + stringFloatEntry.getValue()));
                });

            }
        }

        temp = new ArrayList<>(result);
        result.clear();

        for (int i = 20; i <= 29; i++) {

            float v = Float.parseFloat(indArray.getColumn(i).toString());
            if (v > 0) {
                int finalI = i;
                temp.forEach(stringFloatEntry -> {
                    char[] chars = stringFloatEntry.getKey().toCharArray();
                    chars[2] = jpgLabelList.get(finalI).toCharArray()[2];
                    result.add(new HashMap.SimpleEntry(new String(chars), v + stringFloatEntry.getValue()));
                });

            }
        }

        temp = new ArrayList<>(result);
        result.clear();

        for (int i = 30, len = indArray.columns(); i < len; i++) {

            float v = Float.parseFloat(indArray.getColumn(i).toString());
            if (v > 0) {
                int finalI = i;
                temp.forEach(stringFloatEntry -> {
                    char[] chars = stringFloatEntry.getKey().toCharArray();
                    chars[3] = jpgLabelList.get(finalI).toCharArray()[3];
                    result.add(new HashMap.SimpleEntry(new String(chars), v + stringFloatEntry.getValue()));
                });

            }
        }
        return result.stream().collect(Collectors.toMap(stringFloatEntry -> stringFloatEntry.getKey(), stringFloatEntry -> stringFloatEntry.getValue()));
    }


}
