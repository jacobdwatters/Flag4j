package com.flag4j.operations.concurrency;

import com.flag4j.util.ErrorMessages;

/**
 * Configurations for concurrent operations.
 */
public abstract class Configurations {
    private Configurations() {
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }

    private static final int DEFAULT_NUM_THREADS = Runtime.getRuntime().availableProcessors();
    private static final int DEFAULT_BLOCK_SIZE = 32;
    private static final int DEFAULT_MIN_RECURSIVE_SIZE = 128;

    /**
     * Number of threads to use in concurrent operations.
     */
    private static int numThreads = DEFAULT_NUM_THREADS;

    /**
     * The block size to use in blocked algorithms.
     */
    private static int blockSize = DEFAULT_BLOCK_SIZE;

    /**
     * The minimum size of tensor/matrix/vector to make recursive calls on in recursive algorithms.
     */
    private static int minRecursiveSize = DEFAULT_MIN_RECURSIVE_SIZE;


    /**
     * Sets the number of threads for use in concurrent operations as the number of processors available to the Java
     * virtual machine. Note that this value may change during runtime. This method will include logical cores so the value
     * returned may be higher than the number of physical cores on the machine if hyper-threading is enabled.
     * <br><br>
     * This is implemented as: <code>{@link #numThreads} = {@link Runtime#availableProcessors() Runtime.getRuntime().availableProcessors()};</code>
     * @return The new value of {@link #numThreads}, i.e. the number of available processors.
     */
    static int setNumThreadsAsAvailableProcessors() {
        numThreads = Runtime.getRuntime().availableProcessors();
        return numThreads;
    }


    /**
     * Gets the current number of threads to be used.
     * @return Current number of threads to use in concurrent algorithms.
     */
    public static int getNumThreads() {
        return numThreads;
    }


    /**
     * Sets the number of threads to use in concurrent algorithms.
     * @param numThreads Number of threads to use in concurrent algorithms.
     */
    public static void setNumThreads(int numThreads) {
        Configurations.numThreads = numThreads;
    }


    /**
     * Gets the current block size used in blocked algorithms.
     * @return Current block size to use in concurrent algorithms.
     */
    public static int getBlockSize() {
        return blockSize;
    }


    /**
     * Sets the current block size used in blocked algorithms.
     * @param blockSize Block size to be used in concurrent algorithms.
     */
    public static void setBlockSize(int blockSize) {
        Configurations.blockSize = blockSize;
    }


    /**
     * Gets the minimum size of tensor/matrix/vector to make recursive calls on in recursive algorithms.
     * @return minimum size of tensor/matrix/vector to make recursive calls on in recursive algorithms.
     */
    public static int getMinRecursiveSize() {
        return minRecursiveSize;
    }


    /**
     * Sets the minimum size of tensor/matrix/vector to make recursive calls on in recursive algorithms.
     * @param minRecursiveSize New minimum size.
     */
    public static void setMinRecursiveSize(int minRecursiveSize) {
        Configurations.minRecursiveSize = minRecursiveSize;
    }


    /**
     * Resets all configurations to their default values.
     */
    public static void resetAll() {
        numThreads = DEFAULT_NUM_THREADS;
        blockSize = DEFAULT_BLOCK_SIZE;
        minRecursiveSize = DEFAULT_MIN_RECURSIVE_SIZE;
    }
}
