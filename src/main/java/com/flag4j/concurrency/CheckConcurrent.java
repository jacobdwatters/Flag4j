package com.flag4j.concurrency;

/**
 * A class which contains methods for determining if a concurrent algorithm should be applied.
 */
public final class CheckConcurrent {

    private static final int BASE_THRESHOLD = 9000;
    private static final int LARGER_THRESHOLD = 80000;
    private static final int SMALLER_THRESHOLD = 1000;

    private static final int RELAXED_BASE_THRESHOLD = 4000;
    private static final int RELAXED_LARGER_THRESHOLD = 5000;
    private static final int RELAXED_SMALLER_THRESHOLD = 500;


    /**
     * Applies a simple check to determine if a concurrent algorithm should be applied.
     * @param numRows Number of rows in matrix.
     * @param numCols Number of columns in matrix.
     * @return True if a concurrent algorithm should be used. False if a single thread algorithm should be used.
     */
    public static boolean simpleCheck(int numRows, int numCols) {
        return standardCheck(numRows, numCols, BASE_THRESHOLD, SMALLER_THRESHOLD, LARGER_THRESHOLD);
    }


    /**
     * Applies a check to determine if a concurrent algorithm should be applied. This method is similar to
     * {@link #simpleCheck(int, int)} but uses more relaxed parameters.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return True if a concurrent algorithm should be used. Otherwise, returns false.
     */
    public static boolean relaxedCheck(int numRows, int numCols) {
        return standardCheck(numRows, numCols, RELAXED_BASE_THRESHOLD, RELAXED_SMALLER_THRESHOLD, RELAXED_LARGER_THRESHOLD);
    }


    /**
     * Applies a standard check to determine if a concurrent algorithm should be used for some operation.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @param baseThreshold Base threshold. If both numRows and numCols is greater than this value, then this method
     *                      will return true.
     * @param smallerThreshold Smaller threshold.
     * @param largerThreshold Larger threshold.
     * @return True if a concurrent algorithm should be used.
     */
    private static boolean standardCheck(int numRows, int numCols, int baseThreshold, int smallerThreshold, int largerThreshold) {
        boolean result = false;

        if(Configurations.getNumThreads() > 1) {
            if(numRows >= baseThreshold && numCols >= baseThreshold) {
                result = true;
            } else if(numRows >= largerThreshold && numCols >= smallerThreshold) {
                result = true;
            } else if(numCols >= largerThreshold && numRows >= smallerThreshold) {
                result = true;
            }
        }

        return result;
    }
}
