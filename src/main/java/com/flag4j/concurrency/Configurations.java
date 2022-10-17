package com.flag4j.concurrency;

import com.flag4j.ErrorMessages;

/**
 * Configurations for concurrent operations.
 */
public abstract class Configurations {
    private Configurations() {
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Number of threads to use in concurrent operations.
     */
    static int numThreads = 1;

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
}
