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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.logging.Logger;

/**
 * This class contains the base thread pool for all concurrent operations and several methods for managing the
 * pool.
 */
public final class ThreadManager {
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
    private static ThreadPoolExecutor threadPool = (ThreadPoolExecutor) Executors.newFixedThreadPool(parallelismLevel, daemonFactory);


    /**
     * Sets the number of threads to use in the thread pool.
     * @param parallelismLevel Number of threads to use in the thread pool. If this is less than 1, the parallelism will
     *                         simply be set to 1.
     */
    protected static void setParallelismLevel(int parallelismLevel) {
        ThreadManager.parallelismLevel = Math.max(parallelismLevel, 1);
        threadPool.setCorePoolSize(parallelismLevel);
    }


    /**
     * Gets the current parallelism level for the ThreadManager. That is, the number of threads used in the thread pool.
     * @return The current parallelism level for the ThreadManager.
     */
    public static int getParallelismLevel() {
        return parallelismLevel;
    }


    /**
     * Computes a specified tensor operation concurrently by evenly dividing work amoung available threads (specified by
     * {@link Configurations#getNumThreads()}).
     * @param totalSize Total size of the outer loop for the operation.
     * @param operation Operation to be computed.
     */
    public static void concurrentOperation(final int totalSize, final TensorOperation operation) {
        // Calculate chunk size.
        int chunkSize = (totalSize + parallelismLevel - 1) / parallelismLevel;
        List<Future<?>> futures = new ArrayList<>(parallelismLevel);

        for(int threadIndex = 0; threadIndex < parallelismLevel; threadIndex++) {
            final int startIdx = threadIndex * chunkSize;
            final int endIdx = Math.min(startIdx + chunkSize, totalSize);

            if(startIdx >= endIdx) break; // No more indices to process.

            futures.add(ThreadManager.threadPool.submit(() -> {
                operation.apply(startIdx, endIdx);
            }));
        }

        // Wait for all tasks to complete.
        for(Future<?> future : futures) {
            try {
                future.get(); // Ensure all tasks are complete.
            } catch (InterruptedException | ExecutionException e) {
                // An exception occured.
                threadLogger.warning(e.getMessage());
                Thread.currentThread().interrupt();
            }
        }
    }


    /**
     * Computes a specified blocked tensor operation concurrently by evenly dividing work amoung available threads (specified by
     * {@link Configurations#getNumThreads()}).
     * @param totalSize Total size of the outer loop for the operation.
     * @param blockSize Size of the block used in the blocekdOperation.
     * @param blockedOperation Operation to be computed.
     */
    public static void concurrentBlockedOperation(final int totalSize, final int blockSize, final TensorOperation blockedOperation) {
        // Calculate chunk size for blocks.
        int numBlocks = (totalSize + blockSize - 1) / blockSize;
        List<Future<?>> futures = new ArrayList<>(parallelismLevel);

        for(int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
            final int startBlock = blockIndex * blockSize;
            final int endBlock = Math.min(startBlock + blockSize, totalSize);

            futures.add(threadPool.submit(() -> {
                blockedOperation.apply(startBlock, endBlock);
            }));
        }

        // Wait for all tasks to complete.
        for(Future<?> future : futures) {
            try {
                future.get(); // Ensure all tasks are complete.
            } catch (InterruptedException | ExecutionException e) {
                // An exception occured.
                threadLogger.warning(e.getMessage());
                Thread.currentThread().interrupt();
            }
        }
    }

    // TODO: TEMP FOR TESTING.
    /**
     * Executes a concurrent operation on a given range of indices.
     * The operation is split across multiple threads, each handling a subset of the range.
     *
     * @param totalTasks The total number of tasks (e.g., rows in a matrix) to be processed.
     * @param task A lambda expression or function that takes three arguments: start index, end index, and thread ID.
     *             This function represents the work to be done by each thread for its assigned range.
     */
    public static void concurrentOperation(int totalTasks, TriConsumer<Integer, Integer, Integer> task) {
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        int tasksPerThread = (totalTasks + numThreads - 1) / numThreads;  // Ceiling division

        for (int threadId = 0; threadId < numThreads; threadId++) {
            int startIdx = threadId * tasksPerThread;
            int endIdx = Math.min(startIdx + tasksPerThread, totalTasks);

            if (startIdx < endIdx) {
                final int finalThreadId = threadId;
                executor.submit(() -> task.accept(startIdx, endIdx, finalThreadId));
            }
        }

        executor.shutdown();

        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Thread execution interrupted", e);
        }
    }

    @FunctionalInterface
    public interface TriConsumer<T, U, V> {
        void accept(T t, U u, V v);
    }
}
