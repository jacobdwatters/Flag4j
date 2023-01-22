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

package com.flag4j.util;

import com.flag4j.*;

import java.util.Random;

/**
 * This class contains methods for generating random matrices and vectors.
 */
public class RandomTensor {
    /**
     * Random number generator.
     */
    private final Random rng;
    private final RandomCNumber complex_rng;

    /**
     * Constructs a new random tensor generator with a seed which is unlikely to be the same as other
     * from any other invocation of this constructor.
     */
    public RandomTensor() {
        rng = new Random();
        complex_rng = new RandomCNumber();
    }

    /**
     * Constructs a random tensor generator with a specified seed. Use this constructor for reproducible results.
     * @param seed Seed of the random tensor generator.
     */
    public RandomTensor(long seed) {
        rng = new Random(seed);
        complex_rng = new RandomCNumber(seed);
    }


    /**
     * Gets a tensor with random values. Values are normally distributed with mean of zero and standard deviation
     * of one.
     * @param shape Shape of the tensor.
     * @return A random vector of specified size.
     */
    public Tensor getRandomTensor(Shape shape) {
        Tensor randTensor = new Tensor(shape);

        for(int i=0; i<randTensor.totalEntries().intValue(); i++) {
            randTensor.entries[i] = rng.nextGaussian();
        }

        return randTensor;
    }


    /**
     * Gets a vector with random values. Values are normally distributed with mean of zero and standard deviation
     * of one.
     * @param size Size of the vector.
     * @return A random vector of specified size.
     */
    public Vector getRandomVector(int size) {
        Vector randVec = new Vector(size);

        for(int i=0; i<randVec.totalEntries().intValue(); i++) {
            randVec.entries[i] = rng.nextGaussian();
        }

        return randVec;
    }


    /**
     * Gets a matrix with random values. Values are normally distributed with mean of zero and standard deviation
     * of one.
     * @param rows Number of rows in the resulting matrix.
     * @param cols Number of columns in the resulting matrix.
     * @return A random matrix with specified size.
     */
    public Matrix getRandomMatrix(int rows, int cols) {
        Matrix randMat = new Matrix(rows, cols);

        for(int i=0; i<randMat.totalEntries().intValue(); i++) {
            randMat.entries[i] = rng.nextGaussian();
        }

        return randMat;
    }


    /**
     * Gets a complex tensor with random values. Values are normally distributed with mean of zero and standard deviation
     * of one.
     * @param shape Shape of the tensor.
     * @return A random vector of specified size.
     */
    public CTensor getRandomCTensor(Shape shape) {
        CTensor randTensor = new CTensor(shape);

        for(int i=0; i<randTensor.totalEntries().intValue(); i++) {
            randTensor.entries[i] = complex_rng.randn();
        }

        return randTensor;
    }


    /**
     * Gets a complex matrix with random values. Values are normally distributed with mean of zero and standard deviation
     * of one.
     * @param rows Number of rows in the resulting matrix.
     * @param cols Number of columns in the resulting matrix.
     * @return A random matrix with specified size.
     */
    public CMatrix getRandomCMatrix(int rows, int cols) {
        CMatrix randMat = new CMatrix(rows, cols);

        for(int i=0; i<randMat.totalEntries().intValue(); i++) {
            randMat.entries[i] = complex_rng.randn();
        }

        return randMat;
    }
}
