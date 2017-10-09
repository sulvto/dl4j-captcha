package me.qinchao.captcha;

import org.apache.commons.io.FilenameUtils;

import java.io.File;

/**
 * Created by sulvto on 17-10-6.
 */
public class Const {

    /**
     * root
     */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_captcha");

    /**
     * Location to save and extract the training/testing data
     */
    public static final String TRAINING_PATH = DATA_PATH + File.separator + "captcha_jpg" + File.separator + "training";

    public static final String SAVE_MODEL_FILE = "trained_model.zip";

}
