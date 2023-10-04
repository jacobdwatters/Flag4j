package com.flag4j.complex_numbers;

import com.flag4j.io.PrintOptions;
import com.flag4j.util.ErrorMessages;

/**
 * Contains simple utility functions for the {@link CNumber} object.
 */
public final class CNumberUtils {

    private CNumberUtils() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles.
     * @param src Array for which to compute the max string representation length of a double in.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLength(double[] src) {
        int maxLength = -1;
        int currLength;

        for(double value : src) {
            currLength = CNumber.round(new CNumber(value), PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        return maxLength;
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles.
     * @param src Array for which to compute the max string representation length of a double in.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLength(CNumber[] src) {
        int maxLength = -1;
        int currLength;

        for(CNumber value : src) {
            currLength = CNumber.round(value, PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        return maxLength;
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles up until stopping index.
     * The length of the last element is always considered.
     * @param src Array for which to compute the max string representation length of a double in.
     * @param stopIndex Stopping index for finding max length.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLength(double[] src, int stopIndex) {
        int maxLength = -1;
        int currLength;

        // Ensure no index out of bound exceptions.
        stopIndex = Math.min(stopIndex, src.length);

        for(int i=0; i<stopIndex; i++) {
            currLength = CNumber.round(
                    new CNumber(src[i]),
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        // Always get last elements' length.
        if(stopIndex < src.length) {
            currLength = CNumber.round(
                    new CNumber(src[src.length-1]),
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        return maxLength;
    }


    /**
     * Computes the maximum length of the string representation of a double in an array of doubles up until stopping index.
     * The length of the last element is always considered.
     * @param src Array for which to compute the max string representation length of a double in.
     * @param stopIndex Stopping index for finding max length.
     * @return The maximum length of the string representation of the doubles in the array.
     */
    public static int maxStringLength(CNumber[] src, int stopIndex) {
        int maxLength = -1;
        int currLength;

        // Ensure no index out of bound exceptions.
        stopIndex = Math.min(stopIndex, src.length);

        for(int i=0; i<stopIndex; i++) {
            currLength = CNumber.round(
                    src[i],
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        // Always get last elements' length.
        if(stopIndex < src.length) {
            currLength = CNumber.round(
                    src[src.length-1],
                    PrintOptions.getPrecision()).toString().length();

            if(currLength>maxLength) {
                // Then update the maximum length.
                maxLength = currLength;
            }
        }

        return maxLength;
    }
}
