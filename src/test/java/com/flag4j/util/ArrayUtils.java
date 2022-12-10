package com.flag4j.util;

import com.flag4j.complex_numbers.CNumber;

/**
 * This class provides several methods useful for array manipulation.
 */
public final class ArrayUtils {

    private ArrayUtils() {
        // Hide Constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(int[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(double[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(Integer[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(Double[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Converts array to an array of {@link CNumber complex numbers}.
     * @param array Array to convert.
     * @return An equivalent array of {@link CNumber complex numbers}.
     */
    public static CNumber[] toCNumber(String[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }


    /**
     * Copies an array of {@link CNumber complex numbers}.
     * @param array Array to copy.
     * @return A copy of the specified array.
     */
    public static CNumber[] copyCNumber(CNumber[] array) {
        CNumber[] complexArray = new CNumber[array.length];

        for(int i=0; i<complexArray.length; i++) {
            complexArray[i] = new CNumber(array[i]);
        }

        return complexArray;
    }
}
