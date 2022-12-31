/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j.operations.concurrency;

import com.flag4j.util.ErrorMessages;

import java.util.concurrent.ForkJoinPool;

/**
 * Configurations for standard and concurrent operations.
 */
public abstract class Configurations {
    private Configurations() {
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }

    /**
     * The default number of threads to use for concurrent algorithms.
     */
    private static final int DEFAULT_NUM_THREADS = Runtime.getRuntime().availableProcessors();
    /**
     * The default block size for blocked algorithms.
     */
    private static final int DEFAULT_BLOCK_SIZE = 64;
    /**
     * The default minimum recursive size for recursive algorithms.
     */
    private static final int DEFAULT_MIN_RECURSIVE_SIZE = 128;


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
     * This is implemented as: <code>numThreads = {@link Runtime#availableProcessors() Runtime.getRuntime().availableProcessors()};</code>
     * @return The new value of numThreads, i.e. the number of available processors.
     */
    static int setNumThreadsAsAvailableProcessors() {
        ThreadManager.threadPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        return ThreadManager.threadPool.getParallelism();
    }


    /**
     * Gets the current number of threads to be used.
     * @return Current number of threads to use in concurrent algorithms.
     */
    public static int getNumThreads() {
        return ThreadManager.threadPool.getParallelism();
    }


    /**
     * Sets the number of threads to use in concurrent algorithms.
     * @param numThreads Number of threads to use in concurrent algorithms.
     */
    public static void setNumThreads(int numThreads) {
        ThreadManager.threadPool = new ForkJoinPool(Math.max(1, numThreads));
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
        Configurations.blockSize = Math.max(1, blockSize);
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
        Configurations.minRecursiveSize = Math.max(1, minRecursiveSize);
    }


    /**
     * Resets all configurations to their default values.
     */
    public static void resetAll() {
        ThreadManager.threadPool = new ForkJoinPool(DEFAULT_NUM_THREADS);
        blockSize = DEFAULT_BLOCK_SIZE;
        minRecursiveSize = DEFAULT_MIN_RECURSIVE_SIZE;
    }
}
