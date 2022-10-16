package com.flag4j.concurrency;


/**
 * Configurations for concurrent operations.
 */
public abstract class Configurations {
    /**
     * Number of threads to use in concurrent operations.
     */
    static int NUM_THREADS = 1;


    /**
     * Sets the number of threads for use in concurrent operations as the number of processors available to the Java
     * virtual machine. Note that this value may change during runtime. This method will include logical cores so the value
     * returned may be higher than the number of physical cores on the machine if hyper-threading is enabled.
     * <br><br>
     * This is implemented as: <code>{@link #NUM_THREADS} = {@link Runtime#availableProcessors() Runtime.getRuntime().availableProcessors()};</code>
     * @return The new value of {@link #NUM_THREADS}, i.e. the number of available processors.
     */
    static int setNumThreadsAsAvailableProcessors() {
        NUM_THREADS = Runtime.getRuntime().availableProcessors();
        return NUM_THREADS;
    }
}
