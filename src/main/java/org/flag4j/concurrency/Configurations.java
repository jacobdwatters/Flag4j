/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.concurrency;

import org.flag4j.util.ErrorMessages;

/**
 * Configurations for standard and concurrent operations.
 */
public final class Configurations {
    private Configurations() {
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }

    /**
     * The default number of threads to use for concurrent algorithms.
     */
    public static final int DEFAULT_NUM_THREADS = Runtime.getRuntime().availableProcessors();
    /**
     * The default block size for blocked algorithms.
     */
    public static final int DEFAULT_BLOCK_SIZE = 64;
    /**
     * The default minimum recursive size for recursive algorithms.
     */
    public static final int DEFAULT_MIN_RECURSIVE_SIZE = 128;
    /**
     * The block size to use in blocked algorithms.
     */
    private static int blockSize = DEFAULT_BLOCK_SIZE;


    /**
     * Sets the number of threads for use in concurrent operations as the number of processors available to the Java
     * virtual machine. Note that this value may change during runtime. This method will include logical cores so the value
     * returned may be higher than the number of physical cores on the machine if hyper-threading is enabled.
     * <br><br>
     * @implNote This is implemented as:
     * <code>numThreads = {@link Runtime#availableProcessors() Runtime.getRuntime().availableProcessors()};</code>
     * @return The new value of numThreads, i.e. the number of available processors.
     */
    public static int setNumThreadsAsAvailableProcessors() {
        ThreadManager.setParallelismLevel(Runtime.getRuntime().availableProcessors());
        return ThreadManager.getParallelismLevel();
    }


    /**
     * Gets the current number of threads to be used.
     * @return Current number of threads to use in concurrent algorithms.
     */
    public static int getNumThreads() {
        return ThreadManager.getParallelismLevel();
    }


    /**
     * Sets the number of threads to use in concurrent algorithms.
     * @param numThreads Number of threads to use in concurrent algorithms.
     */
    public static void setNumThreads(int numThreads) {
        ThreadManager.setParallelismLevel(numThreads);
    }


    /**
     * Gets the current block size used in blocked algorithms. If it has not been changed it will {@link #DEFAULT_BLOCK_SIZE default to 64}.
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
     * Resets all configurations to their default values.
     */
    public static void resetAll() {
        ThreadManager.setParallelismLevel(DEFAULT_NUM_THREADS);
        blockSize = DEFAULT_BLOCK_SIZE;
    }
}
