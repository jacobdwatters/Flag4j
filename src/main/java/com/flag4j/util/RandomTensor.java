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
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.Decompose;

/**
 * This class contains methods for generating pseudorandom tensors, matrices, and vectors.
 */
public class RandomTensor {
    /**
     * Complex pseudorandom number generator.
     */
    private final RandomCNumber complexRng;

    /**
     * Constructs a new pseudorandom tensor generator with a seed which is unlikely to be the same as other
     * from any other invocation of this constructor.
     */
    public RandomTensor() {
        complexRng = new RandomCNumber();
    }

    /**
     * Constructs a pseudorandom tensor generator with a specified seed. Use this constructor for reproducible results.
     * @param seed Seed of the pseudorandom tensor generator.
     */
    public RandomTensor(long seed) {
        complexRng = new RandomCNumber(seed);
    }


    /**
     * Generates a tensor filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param shape Shape of the tensor.
     * @return A tensor filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Tensor randomTensor(Shape shape) {
        return new Tensor(shape, genUniformRealArray(shape.totalEntries().intValueExact()));
    }


    /**
     * Generates a tensor filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param shape Shape of the tensor.
     * @param min Minimum value for the uniform distribution.
     * @param max Maximum value for the uniform distribution.
     * @return A tensor filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     */
    public Tensor randomTensor(Shape shape, double min, double max) {
        return new Tensor(shape, genUniformRealArray(
                shape.totalEntries().intValueExact(),
                min,
                max)
        );
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     * @param shape Shape of the tensor.
     * @return A tensor filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     */
    public Tensor randnTensor(Shape shape) {
        return new Tensor(shape, genNormalRealArray(shape.totalEntries().intValueExact()));
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @param shape Shape of the tensor.
     * @param mean Mean of the normal distribution to sample from.
     * @param std Standard deviation of normal distribution to sample from.
     * @return A tensor filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public Tensor randnTensor(Shape shape, double mean, double std) {
        return new Tensor(shape, genNormalRealArray(
                shape.totalEntries().intValueExact(),
                mean,
                std)
        );
    }


    /**
     * Generates a tensor filled with pseudorandom complex values with magnitudes
     * uniformly distributed in {@code [0, 1)}.
     * @param shape Shape of the tensor.
     * @return A tensor filled with pseudorandom complex values with magnitudes
     * uniformly distributed in {@code [0, 1)}.
     */
    public CTensor randomCTensor(Shape shape) {
        return new CTensor(shape, genUniformComplexArray(shape.totalEntries().intValueExact()));
    }


    /**
     * Generates a tensor filled with pseudorandom values with magnitudes uniformly distributed in {@code [min, max)}.
     * @param shape Shape of the tensor.
     * @param min Minimum value for the uniform distribution from which to sample magnitude.
     * @param max Maximum value for the uniform distribution from which to sample magnitude.
     * @return A tensor filled with pseudorandom values with magnitudes uniformly distributed in {@code [min, max)}
     * @throws IllegalArgumentException If {@code min} is negative or if {@code max} is less than {@code min}.
     */
    public CTensor randomCTensor(Shape shape, double min, double max) {
        return new CTensor(shape, genUniformComplexArray(
                shape.totalEntries().intValueExact(),
                min,
                max)
        );
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     * @param shape Shape of the tensor.
     * @return A tensor filled with pseudorandom values sampled from a normal distribution with
     * mean of 0.0 and standard deviation of 1.0.
     */
    public CTensor randnCTensor(Shape shape) {
        return new CTensor(shape, genNormalComplexArray(shape.totalEntries().intValueExact()));
    }


    /**
     * Generates a tensor filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @param shape Shape of the tensor.
     * @param mean Mean of the normal distribution to sample from.
     * @param std Standard deviation of normal distribution to sample from.
     * @return A tensor filled with pseudorandom values sampled from a normal distribution with
     * specified mean and standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public CTensor randnCTensor(Shape shape, double mean, double std) {
        return new CTensor(shape, genNormalComplexArray(
                shape.totalEntries().intValueExact(),
                mean,
                std)
        );
    }


    /**
     * Generates a vector filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Vector randomVector(int size) {
        return new Vector(genUniformRealArray(size));
    }



    /**
     * Generates a vector filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     */
    public Vector randomVector(int size, double min, double max) {
        return new Vector(genUniformRealArray(size, min, max));
    }


    /**
     * Generates a vector filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public Vector randnVector(int size) {
        return new Vector(genNormalRealArray(size));
    }


    /**
     * Generates a vector filled with pseudorandom values sampled from a normal distribution with specified mean
     * and standard deviation.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values sampled from a normal distribution with specified mean
     * and standard deviation.
     * @throws IllegalArgumentException If the standard deviation is negative.
     */
    public Vector randnVector(int size, double mean, double std) {
        return new Vector(genNormalRealArray(size));
    }



    /**
     * Generates a vector filled with pseudorandom complex values with magnitudes uniformly distributed in {@code [0, 1)}.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom complex values with magnitudes uniformly distributed in {@code [0, 1)}.
     */
    public CVector randomCVector(int size) {
        return new CVector(genUniformComplexArray(size));
    }



    /**
     * Generates a vector filled with pseudorandom values with magnitudes uniformly distributed in {@code [min, max)}.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values with magnitudes uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code min} is negative or if {@code max} is less than {@code min}.
     */
    public CVector randomCVector(int size, double min, double max) {
        return new CVector(genUniformComplexArray(size, min, max));
    }


    /**
     * Generates a vector filled with pseudorandom values with magnitudes sampled from a normal distribution with a
     * mean of 0.0 and a standard deviation of 1.0.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values with magnitudes sampled from a normal distribution with a
     * mean of 0.0 and a standard deviation of 1.0.
     */
    public CVector randnCVector(int size) {
        return new CVector(genNormalComplexArray(size));
    }


    /**
     * Generates a vector filled with pseudorandom values with magnitudes sampled from a normal distribution with specified mean
     * and standard deviation.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values with magnitudes sampled from a normal distribution with specified mean
     *      * and standard deviation.
     * @throws IllegalArgumentException If the standard deviation is negative.
     */
    public CVector randnCVector(int size, double mean, double std) {
        return new CVector(genNormalComplexArray(size, mean, std));
    }


    /**
     * Generates a matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param rows Number of rows in the resulting matrix.
     * @param cols Number of columns in the resulting matrix.
     * @return A matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Matrix randomMatrix(int rows, int cols) {
        return new Matrix(rows, cols, genUniformRealArray(rows*cols));
    }


    /**
     * Generates a matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param shape Shape of the resulting matrix. Must be of rank 2.
     * @return A matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix randomMatrix(Shape shape) {
        return randomMatrix(shape.get(0), shape.get(1));
    }


    /**
     * Generates a matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param rows Number of rows in the resulting matrix.
     * @param cols Number of columns in the resulting matrix.
     * @param min Minimum value of uniform distribution to sample from (inclusive).
     * @param max Maximum value of uniform distribution to sample from (exclusive).
     * @return A matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     */
    public Matrix randomMatrix(int rows, int cols, double min, double max) {
        return new Matrix(rows, cols, genUniformRealArray(rows*cols, min, max));
    }


    /**
     * Generates a matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param shape Shape of the resulting matrix. Must be of rank 2.
     * @param min Minimum value of uniform distribution to sample from (inclusive).
     * @param max Maximum value of uniform distribution to sample from (exclusive).
     * @return A matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     * @throws IllegalArgumentException If {@code shape} is not of rank 2.
     */
    public Matrix randomMatrix(Shape shape, double min, double max) {
        return randomMatrix(shape.get(0), shape.get(1), min, max);
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param rows The number of rows in the resulting matrix.
     * @param cols The number of columns in the resulting matrix.
     * @return A matrix filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public Matrix randnMatrix(int rows, int cols) {
        return new Matrix(rows, cols, genNormalRealArray(rows*cols));
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param shape Shape of the resulting matrix.
     * @return A matrix filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public Matrix randnMatrix(Shape shape) {
        return randnMatrix(shape.get(0), shape.get(1));
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a normal distribution with a specified mean
     * and standard deviation.
     * @param rows The number of rows in the resulting matrix.
     * @param cols The number of columns in the resulting matrix.
     * @param mean Mean of the normal distribution from which values are sampled.
     * @param std Standard deviation of the normal distribution from which values are sampled.
     * @return A matrix filled with pseudorandom values sampled from a normal distribution with a specified mean
     * and standard deviation.
     * @throws IllegalArgumentException If the standard deviation is negative.
     */
    public Matrix randnMatrix(int rows, int cols, double mean, double std) {
        return new Matrix(rows, cols, genNormalRealArray(rows*cols, mean, std));
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a normal distribution with a specified mean
     * and standard deviation.
     * @param shape Shape of the resulting matrix.
     * @param mean Mean of the normal distribution from which values are sampled.
     * @param std Standard deviation of the normal distribution from which values are sampled.
     * @return A matrix filled with pseudorandom values sampled from a normal distribution with a specified mean
     * and standard deviation.
     * @throws IllegalArgumentException If the standard deviation is negative.
     */
    public Matrix randnMatrix(Shape shape, double mean, double std) {
        return randnMatrix(shape.get(0), shape.get(1), mean, std);
    }


    /**
     * Generates a symmetric matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Matrix getRandomSymmetricMatrix(int size) {
        Matrix randMat = new Matrix(size);

        for(int i=0; i<size; i++) {
            for(int j=0; j<i; j++) {
                randMat.entries[i*size+j] = complexRng.rng.nextDouble();
                randMat.entries[j*size+i] = randMat.entries[i*size+j];
            }

            randMat.entries[i*(size+1)] = complexRng.rng.nextDouble(); // Diagonal entry
        }

        return randMat;
    }


    /**
     * Gets a pseudorandom orthogonal matrix. From an implementation point of view, a pseudorandom matrix is generated
     * as if by {@link #randomMatrix(int, int) getRandomMatrix(size, size)}. Then, a {@link Decompose#QR(Matrix) QR}
     * decomposition is computed on this pseudorandom matrix and the {@code Q} matrix from this decomposition is returned.
     * @param size Size of the orthogonal matrix (i.e. the number rows and columns for the square matrix).
     * @return A pseudorandom orthogonal matrix.
     */
    public Matrix randomOrthogonalMatrix(int size) {
        Matrix randMat = new Matrix(size, size, genUniformRealArray(size));
        return Decompose.QR(randMat)[0];
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes uniformly
     * distributed in {@code [0, 1)}.
     * @param rows Number of rows in the resulting matrix.
     * @param cols Number of columns in the resulting matrix.
     * @return A matrix filled with pseudorandom complex values with magnitudes uniformly
     * distributed in {@code [0, 1)}.
     */
    public CMatrix randomCMatrix(int rows, int cols) {
        return new CMatrix(rows, cols, genUniformComplexArray(rows*cols));
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes uniformly
     * distributed in {@code [0, 1)}.
     * @param shape Shape of the resulting matrix. Must be of rank 2.
     * @return A matrix filled with pseudorandom complex values with magnitudes uniformly
     * distributed in {@code [0, 1)}.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public CMatrix randomCMatrix(Shape shape) {
        return randomCMatrix(shape.get(0), shape.get(1));
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes
     * uniformly distributed in {@code [min, max)}.
     * @param rows Number of rows in the resulting matrix.
     * @param cols Number of columns in the resulting matrix.
     * @param min Minimum value of uniform distribution to sample from (inclusive).
     * @param max Maximum value of uniform distribution to sample from (exclusive).
     * @return A matrix filled with pseudorandom complex values with magnitudes
     * uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     */
    public CMatrix randomCMatrix(int rows, int cols, double min, double max) {
        return new CMatrix(rows, cols, genUniformComplexArray(rows*cols, min, max));
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes
     * uniformly distributed in {@code [min, max)}.
     * @param shape Shape of the resulting matrix. Must be of rank 2.
     * @param min Minimum value of uniform distribution to sample from (inclusive).
     * @param max Maximum value of uniform distribution to sample from (exclusive).
     * @return A matrix filled with pseudorandom complex values with magnitudes
     * uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     * @throws IllegalArgumentException If {@code shape} is not of rank 2.
     */
    public CMatrix randomCMatrix(Shape shape, double min, double max) {
        return randomCMatrix(shape.get(0), shape.get(1), min, max);
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param rows The number of rows in the resulting matrix.
     * @param cols The number of columns in the resulting matrix.
     * @return A matrix filled with pseudorandom complex values with magnitudes sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public CMatrix randnCMatrix(int rows, int cols) {
        return new CMatrix(rows, cols, genNormalComplexArray(rows*cols));
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param shape Shape of the resulting matrix.
     * @return A matrix filled with pseudorandom complex values with magnitudes sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public CMatrix randnCMatrix(Shape shape) {
        return randnCMatrix(shape.get(0), shape.get(1));
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes sampled from a normal
     * distribution with a specified mean and standard deviation.
     * @param rows The number of rows in the resulting matrix.
     * @param cols The number of columns in the resulting matrix.
     * @param mean Mean of the normal distribution from which values are sampled.
     * @param std Standard deviation of the normal distribution from which values are sampled.
     * @return A matrix filled with pseudorandom complex values with magnitudes sampled from a normal
     * distribution with a specified mean and standard deviation.
     * @throws IllegalArgumentException If the standard deviation is negative.
     */
    public CMatrix randnCMatrix(int rows, int cols, double mean, double std) {
        return new CMatrix(rows, cols, genNormalComplexArray(rows*cols, mean, std));
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a normal distribution with a specified mean
     * and standard deviation.
     * @param shape Shape of the resulting matrix.
     * @param mean Mean of the normal distribution from which values are sampled.
     * @param std Standard deviation of the normal distribution from which values are sampled.
     * @return A matrix filled with pseudorandom values sampled from a normal distribution with a specified mean
     * and standard deviation.
     * @throws IllegalArgumentException If the standard deviation is negative.
     */
    public CMatrix randnCMatrix(Shape shape, double mean, double std) {
        return randnCMatrix(shape.get(0), shape.get(1), mean, std);
    }

    // TODO: Once the QR decomposition is implemented for complex matrices, add randomUnitaryMatrix(int size) method.

    /**
     * Generates an array of doubles filled with uniformly distributed pseudorandom values in {@code [0.0, 1.0)}.
     * To generate uniformly distributed values in a specified range see {@link #genUniformRealArray(int, double, double)}.
     * @param length Length of pseudorandom array to generate.
     * @return An array of doubles with specified length filled with uniformly distributed values in {@code [0.0, 1.0)}.
     */
    private double[] genUniformRealArray(int length) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.rng.nextDouble();
        }

        return values;
    }


    /**
     * Generates an array of doubles filled with uniformly distributed pseudorandom values in {@code [min, max)}.
     * @param length Length of pseudorandom array to generate.
     * @param min Lower bound of uniform range (inclusive).
     * @param max Upper bound of uniform range (Exclusive).
     * @return An array of doubles with specified length filled with uniformly distributed values in {@code [min, max)}.
     */
    private double[] genUniformRealArray(int length, double min, double max) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.rng.nextDouble()*(max - min) + min;
        }

        return values;
    }


    /**
     * Generates an array of integers filled with uniformly distributed pseudorandom values in {@code [0, 1)}.
     * To generate uniformly distributed values in a specified range see {@link #genUniformRealIntArray(int, int, int)}.
     * @param length Length of pseudorandom array to generate.
     * @return An array of integers with specified length filled with uniformly distributed values in {@code [0, 1)}.
     */
    private double[] genUniformRealIntArray(int length) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.rng.nextInt();
        }

        return values;
    }


    /**
     * Generates an array of integers filled with uniformly distributed pseudorandom values in {@code [min, max)}.
     * @param length Length of pseudorandom array to generate.
     * @param min Lower bound of uniform range (inclusive).
     * @param max Upper bound of uniform range (Exclusive).
     * @return An array of integers with specified length filled with uniformly distributed values in {@code [min, max)}.
     */
    private double[] genUniformRealIntArray(int length, int min, int max) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.rng.nextInt()*(max - min) + min;
        }

        return values;
    }


    /**
     * Generates an array of doubles filled with normally pseudorandom values with a mean of 0 and standard deviation
     * of 1.
     * @param length Length of pseudorandom array to generate.
     * @return An array of doubles with specified length filled with normally pseudorandom values with a mean of 0 and
     * standard deviation of 1.
     */
    private double[] genNormalRealArray(int length) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.rng.nextGaussian();
        }

        return values;
    }


    /**
     * Generates an array of doubles filled with normally distributed pseudorandom values with a specified mean and standard deviation.
     * @param length Length of pseudorandom array to generate.
     * @param mean Mean of normal distribution.
     * @param std Standard deviation of normal distribution.
     * @return An array of doubles with specified length filled with normally pseudorandom values specified mean and
     * standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    private double[] genNormalRealArray(int length, double mean, double std) {
        double[] values = new double[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.rng.nextGaussian()*mean + std;
        }

        return values;
    }


    /**
     * Generates an array of {@link CNumber complex numbers} with pseudorandom uniformly distributed magnitudes
     * in {@code [0.0, 1.0)}.
     * @param length Length of the pseudorandom array to generate.
     * @return An array of {@link CNumber complex numbers} with pseudorandom uniformly distributed magnitudes
     * in {@code [0.0, 1.0)}.
     */
    private CNumber[] genUniformComplexArray(int length) {
        CNumber[] values = new CNumber[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.random();
        }

        return values;
    }


    /**
     * Generates an array of pseudorandom {@link CNumber complex numbers} with uniformly distributed magnitudes
     * in {@code [min, max)}.
     * @param length Length of the pseudorandom array to generate.
     * @param min Minimum value of uniform distribution from which the magnitudes are sampled (inclusive).
     * @param max Maximum value of uniform distribution from which the magnitudes are sampled (exclusive).
     * @return An array of {@link CNumber complex numbers} with pseudorandom {@link CNumber complex numbers} with
     * uniformly distributed magnitudes in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code min} is negative or if {@code max} is less than {@code min}.
     */
    private CNumber[] genUniformComplexArray(int length, double min, double max) {
        CNumber[] values = new CNumber[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.random(min, max);
        }

        return values;
    }


    /**
     * Generates an array of {@link CNumber complex numbers} with pseudorandom normally distributed magnitudes
     * with a mean of 0.0 and a magnitude of 1.0.
     * @param length Length of the pseudorandom array to generate.
     * @return An array of {@link CNumber complex numbers} with pseudorandom normally distributed magnitudes
     * with a mean of 0.0 and a magnitude of 1.0.
     */
    private CNumber[] genNormalComplexArray(int length) {
        CNumber[] values = new CNumber[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.randn();
        }

        return values;
    }


    /**
     * Generates an array of {@link CNumber complex numbers} with pseudorandom normally distributed magnitudes
     * with specified mean and standard deviation.
     * @param length Length of the pseudorandom array to generate.
     * @param mean Mean of the normal distribution from which to sample magnitudes.
     * @param std Standard deviation of the normal distribution from which to sample magnitudes.
     * @return An array of {@link CNumber complex numbers} with pseudorandom normally distributed magnitudes
     * with specified mean and standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    private CNumber[] genNormalComplexArray(int length, double mean, double std) {
        CNumber[] values = new CNumber[length];

        for(int i=0; i<length; i++) {
            values[i] = complexRng.randn(mean, std);
        }

        return values;
    }
}
