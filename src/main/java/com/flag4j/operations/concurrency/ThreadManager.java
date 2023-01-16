/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

import com.flag4j.operations.concurrency.util.ErrorMessages;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/**
 * This class contains the base thread pool for all concurrent operations and several methods for managing the
 * pool.
 */
public class ThreadManager {

    /**
     * Thread pool for managing threads executing concurrent operations.
     */
    protected static ForkJoinPool threadPool = new ForkJoinPool();


    /**
     * Applies a concurrent loop to a function.
     * @param startIndex Starting index for concurrent loop.
     * @param endIndex Ending index for concurrent loop (exclusive).
     * @param function Function to apply each iteration. Function is dependent on iteration index.
     */
    public static void concurrentLoop(int startIndex, int endIndex, IntConsumer function) {
        try {
            threadPool.submit(() -> IntStream.range(startIndex, endIndex).parallel().forEach(function)).get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            Thread.currentThread().interrupt();
        }
    }



    /**
     * Applies a concurrent loop to a function.
     * @param startIndex Starting index for concurrent loop.
     * @param endIndex Ending index for concurrent loop (exclusive).
     * @param step Step size for the index variable of the loop.
     * @param function Function to apply each iteration. Function is dependent on iteration index.
     */
    public static void concurrentLoop(int startIndex, int endIndex, int step, IntConsumer function ) {
        if (step <= 0)
            throw new IllegalArgumentException(ErrorMessages.negValueErr(startIndex));
        try {
            int range = endIndex - startIndex;
            int iterations = range/step + ((range%step == 0) ? 0 : 1);
            threadPool.submit(() -> IntStream.range(0, iterations).parallel().forEach(
                    i -> function.accept(startIndex + i*step))
            ).get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            Thread.currentThread().interrupt();
        }
    }
}
