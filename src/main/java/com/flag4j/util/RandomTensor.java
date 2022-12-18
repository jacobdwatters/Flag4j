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

package com.flag4j.util;

import com.flag4j.Matrix;

import java.util.Random;

/**
 * This class contains methods for generating random matrices and vectors.
 */
public class RandomTensor {
    private final Random rng;

    public RandomTensor() {
        rng = new Random();
    }

    public RandomTensor(long seed) {
        rng = new Random(seed);
    }


    /**
     * Gets a matrix with random values. Values are normally distributed with mean of zero and standard deviation
     * of one.
     * @param m Number of rows in the resulting matrix.
     * @param n Number of columns in the resulting matrix.
     * @return A random matrix with specified size.
     */
    public Matrix getRandomMatrix(int m, int n) {
        Matrix randMat = new Matrix(m, n);

        for(int i=0; i<randMat.totalEntries().intValue(); i++) {
            randMat.entries[i] = rng.nextGaussian();
        }

        return randMat;
    }
}
