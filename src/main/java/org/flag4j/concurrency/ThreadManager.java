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

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.function.IntConsumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * This class contains the base thread pool for all concurrent operations and several methods for managing the
 * pool.
 */
public class ThreadManager {
    private ThreadManager() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }

    /**
     * Simple thread factory for creating basic daemon threads.
     */
    private static final ThreadFactory daemonFactory = r -> {
        Thread t = new Thread(r);
        t.setDaemon(true); // Set the thread as a daemon thread
        return t;
    };

    /**
     * The parallelism level for the thread manager. That is, the number of threads to be used in the thread pool
     * when executing concurrent operations.
     */
    private static int parallelismLevel = Configurations.DEFAULT_NUM_THREADS;

    /**
     * Simple logger for when a thread throws an exception during execution.
     */
    private static final Logger threadLogger = Logger.getLogger(ThreadManager.class.getName());

    /**
     * Thread pool for managing threads executing concurrent operations.
     */
    protected static ExecutorService threadPool = Executors.newFixedThreadPool(parallelismLevel, daemonFactory);


    /**
     * Sets the number of threads to use in the thread pool.
     * @param parallelismLevel Number of threads to use in the thread pool. If this is less than 1, the parallelism will
     *                         simply be set to 1.
     */
    protected static void setParallelismLevel(int parallelismLevel) {
        ThreadManager.parallelismLevel = Math.max(parallelismLevel, 1);
        threadPool = Executors.newFixedThreadPool(parallelismLevel, daemonFactory);
    }


    /**
     * Gets the current parallelism level for the ThreadManager. That is, the number of threads used in the thread pool.
     * @return The current parallelism level for the ThreadManager.
     */
    public static int getParallelismLevel() {
        return parallelismLevel;
    }


    /**
     * Applies a concurrent loop to a function.
     * @param startIndex Starting index for concurrent loop (inclusive).
     * @param endIndex Ending index for concurrent loop (exclusive).
     * @param function Function to apply each iteration. Function may be dependent on iteration index but should
     *                 individual iterations should be independent of each other.
     */
    public static void concurrentLoop(int startIndex, int endIndex, IntConsumer function) {
        try {
            threadPool.submit(() -> IntStream.range(startIndex, endIndex).parallel().forEach(function)).get();
        } catch (InterruptedException | ExecutionException e) {
            threadLogger.setLevel(Level.WARNING);
            threadLogger.warning(e.getMessage());
            Thread.currentThread().interrupt();
        }
    }


    /**
     * Applies a concurrent strided-loop to a function.
     * @param startIndex Starting index for concurrent loop (inclusive).
     * @param endIndex Ending index for concurrent loop (exclusive).
     * @param step Step size for the index variable of the loop (i.e. the stride size).
     * @param function Function to apply each iteration. Function may be dependent on iteration index but should
     *      individual iterations should be independent of each other.
     */
    public static void concurrentLoop(int startIndex, int endIndex, int step, IntConsumer function) {
        if(step <= 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(startIndex));
        try {
            int range = endIndex - startIndex;
            int iterations = range/step + ((range%step == 0) ? 0 : 1);
            threadPool.submit(() -> IntStream.range(0, iterations).parallel().forEach(
                    i -> function.accept(startIndex + i*step))
            ).get();
        } catch (InterruptedException | ExecutionException e) {
            threadLogger.setLevel(Level.WARNING);
            threadLogger.warning(e.getMessage());
            Thread.currentThread().interrupt();
        }
    }
}
