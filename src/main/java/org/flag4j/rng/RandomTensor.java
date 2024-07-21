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

package org.flag4j.rng;

import org.flag4j.arrays.dense.*;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.linalg.decompositions.qr.ComplexQR;
import org.flag4j.linalg.decompositions.qr.RealQR;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * An instance of this class is used for generating streams of pseudorandom tensors, matrices, and vectors.
 */
public class RandomTensor {
    /**
     * Complex pseudorandom number generator.
     */
    private final RandomCNumber COMPLEX_RNG;
    private final RandomArray RAND_ARRAY;

    /**
     * Constructs a new pseudorandom tensor generator with a seed which is unlikely to be the same as other
     * from any other invocation of this constructor.
     */
    public RandomTensor() {
        COMPLEX_RNG = new RandomCNumber();
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
    }

    /**
     * Constructs a pseudorandom tensor generator with a specified seed. Use this constructor for reproducible results.
     * @param seed Seed of the pseudorandom tensor generator.
     */
    public RandomTensor(long seed) {
        COMPLEX_RNG = new RandomCNumber(seed);
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
    }


    /**
     * Generates a tensor filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param shape Shape of the tensor.
     * @return A tensor filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Tensor randomTensor(Shape shape) {
        return new Tensor(shape, RAND_ARRAY.genUniformRealArray(shape.totalEntries().intValueExact()));
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
        return new Tensor(shape, RAND_ARRAY.genUniformRealArray(
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
        return new Tensor(shape, RAND_ARRAY.genNormalRealArray(shape.totalEntries().intValueExact()));
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
        return new Tensor(shape, RAND_ARRAY.genNormalRealArray(
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
        return new CTensor(shape, RAND_ARRAY.genUniformComplexArray(shape.totalEntries().intValueExact()));
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
        return new CTensor(shape, RAND_ARRAY.genUniformComplexArray(
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
        return new CTensor(shape, RAND_ARRAY.genNormalComplexArray(shape.totalEntries().intValueExact()));
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
        return new CTensor(shape, RAND_ARRAY.genNormalComplexArray(
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
        return new Vector(RAND_ARRAY.genUniformRealArray(size));
    }



    /**
     * Generates a vector filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code max} is less than {@code min}.
     */
    public Vector randomVector(int size, double min, double max) {
        return new Vector(RAND_ARRAY.genUniformRealArray(size, min, max));
    }


    /**
     * Generates a vector filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public Vector randnVector(int size) {
        return new Vector(RAND_ARRAY.genNormalRealArray(size));
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
        return new Vector(RAND_ARRAY.genNormalRealArray(size, mean, std));
    }



    /**
     * Generates a vector filled with pseudorandom complex values with magnitudes uniformly distributed in {@code [0, 1)}.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom complex values with magnitudes uniformly distributed in {@code [0, 1)}.
     */
    public CVector randomCVector(int size) {
        return new CVector(RAND_ARRAY.genUniformComplexArray(size));
    }



    /**
     * Generates a vector filled with pseudorandom values with magnitudes uniformly distributed in {@code [min, max)}.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values with magnitudes uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code min} is negative or if {@code max} is less than {@code min}.
     */
    public CVector randomCVector(int size, double min, double max) {
        return new CVector(RAND_ARRAY.genUniformComplexArray(size, min, max));
    }


    /**
     * Generates a vector filled with pseudorandom values with magnitudes sampled from a normal distribution with a
     * mean of 0.0 and a standard deviation of 1.0.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values with magnitudes sampled from a normal distribution with a
     * mean of 0.0 and a standard deviation of 1.0.
     */
    public CVector randnCVector(int size) {
        return new CVector(RAND_ARRAY.genNormalComplexArray(size));
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
        return new CVector(RAND_ARRAY.genNormalComplexArray(size, mean, std));
    }


    /**
     * Generates a matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param rows Number of rows in the resulting matrix.
     * @param cols Number of columns in the resulting matrix.
     * @return A matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Matrix randomMatrix(int rows, int cols) {
        return new Matrix(rows, cols, RAND_ARRAY.genUniformRealArray(rows*cols));
    }


    /**
     * Generates a matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param shape Shape of the resulting matrix. Must be of rank 2.
     * @return A matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix randomMatrix(Shape shape) {
        ParameterChecks.assertRank(2, shape);
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
        return new Matrix(rows, cols, RAND_ARRAY.genUniformRealArray(rows*cols, min, max));
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
        ParameterChecks.assertRank(2, shape);
        return randomMatrix(shape.get(0), shape.get(1), min, max);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(int rows, int cols, double min, double max, double sparsity) {
        return randomCooMatrix(new Shape(rows, cols), min, max, sparsity);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(Shape shape, double min, double max, double sparsity) {
        ParameterChecks.assertRank(2, shape);

        int numEntries = new BigDecimal(shape.totalEntries()).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();

        return randomCooMatrix(shape, min, max, numEntries);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero entries. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero entries int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero entries uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(int rows, int cols, double min, double max, int numNonZeroEntries) {
        return randomCooMatrix(new Shape(rows, cols), min, max, numNonZeroEntries);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero entries. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero entries int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero entries uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(Shape shape, double min, double max, int numNonZeroEntries) {
        ParameterChecks.assertGreaterEq(0, numNonZeroEntries);
        ParameterChecks.assertLessEq(shape.totalEntries(), numNonZeroEntries, "numNonZeroEntries");

        double[] entries = RAND_ARRAY.genUniformRealArray(numNonZeroEntries, min, max);
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numNonZeroEntries, 0, shape.get(0), 0, shape.get(1));

        return new CooMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(int rows, int cols, double min, double max, double sparsity) {
        return randomCooMatrix(new Shape(rows, cols), min, max, sparsity).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(Shape shape, double min, double max, double sparsity) {
        ParameterChecks.assertRank(2, shape);

        int numEntries = new BigDecimal(shape.totalEntries()).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();

        return randomCooMatrix(shape, min, max, numEntries).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero entries. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero entries int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero entries uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(int rows, int cols, double min, double max, int numNonZeroEntries) {
        return randomCooMatrix(new Shape(rows, cols), min, max, numNonZeroEntries).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero entries. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero entries int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero entries uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(Shape shape, double min, double max, int numNonZeroEntries) {
        ParameterChecks.assertGreaterEq(0, numNonZeroEntries);
        ParameterChecks.assertLessEq(shape.totalEntries(), numNonZeroEntries, "numNonZeroEntries");

        double[] entries = RAND_ARRAY.genUniformRealArray(numNonZeroEntries, min, max);
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numNonZeroEntries, 0, shape.get(0), 0, shape.get(1));

        return new CooMatrix(shape, entries, indices[0], indices[1]).toCsr();
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
        return new Matrix(rows, cols, RAND_ARRAY.genNormalRealArray(rows*cols));
    }


    /**
     * Generates a matrix filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param shape Shape of the resulting matrix.
     * @return A matrix filled with pseudorandom values sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public Matrix randnMatrix(Shape shape) {
        ParameterChecks.assertRank(2, shape);
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
        return new Matrix(rows, cols, RAND_ARRAY.genNormalRealArray(rows*cols, mean, std));
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
        ParameterChecks.assertRank(2, shape);
        return randnMatrix(shape.get(0), shape.get(1), mean, std);
    }


    /**
     * Generates a symmetric matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [0, 1)}.
     */
    public Matrix randomSymmetricMatrix(int size) {
        Matrix randMat = new Matrix(size);

        for(int i=0; i<size; i++) {
            for(int j=0; j<i; j++) {
                randMat.entries[i*size+j] = COMPLEX_RNG.nextDouble();
                randMat.entries[j*size+i] = randMat.entries[i*size+j];
            }

            randMat.entries[i*(size+1)] = COMPLEX_RNG.nextDouble(); // Diagonal entry
        }

        return randMat;
    }


    /**
     * Generates a symmetric {@link CooMatrix COO matrix} filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @param min Minimum value in uniform distribution.
     * @param max Maximum value in uniform distribution.
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     * a value in {@code [0.0, 1.0]}. The true sparsity may slightly differ to ensure the matrix is symmetric.
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code sparsity} is not in the range {@code [0.0, 1.0]}.
     */
    public CooMatrix randomSymmetricCooMatrix(int size, int min, int max, double sparsity) {
        ParameterChecks.assertInRange(sparsity, 0, 1, "sparsity");
        Shape shape = new Shape(size, size);

        int numEntries = new BigDecimal(size).pow(2).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();
        numEntries /= 2;

        // Generate half of the random entries.
        double[] entries = RAND_ARRAY.genUniformRealArray(numEntries, min, max);
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numEntries, 0, shape.get(0), 0, shape.get(1));

        // Mirror entries across diagonal.
        entries = ArrayUtils.join(entries, entries);
        indices = new int[][]{
                ArrayUtils.join(indices[0], indices[1]),
                ArrayUtils.join(indices[1], indices[0])
        };

        CooMatrix randMat = new CooMatrix(shape, entries, indices[0], indices[1]);
        randMat.sortIndices();

        return new CooMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Generates a symmetric {@link CsrMatrix CSR matrix} filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @param size Number of rows and columns in the resulting matrix (the result will be a square matrix).
     * @param min Minimum value in uniform distribution.
     * @param max Maximum value in uniform distribution.
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     * a value in {@code [0.0, 1.0]}. The true sparsity may slightly differ to ensure the matrix is symmetric.
     * @return A symmetric matrix filled with pseudorandom values uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code sparsity} is not in the range {@code [0.0, 1.0]}.
     */
    public CsrMatrix randomSymmetricCsrMatrix(int size, int min, int max, double sparsity) {
        return randomSymmetricCooMatrix(size, min, max, sparsity).toCsr();
    }


    /**
     * Gets a pseudorandom orthogonal matrix. From an implementation point of view, a pseudorandom matrix is generated
     * as if by {@link #randomMatrix(int, int) randomMatrix(size, size)}. Then, a {@link RealQR QR}
     * decomposition is computed on this pseudorandom matrix and the {@code Q} matrix from this decomposition is returned.
     * @param size Size of the orthogonal matrix (i.e. the number rows and columns for the square matrix).
     * @return A pseudorandom orthogonal matrix.
     */
    public Matrix randomOrthogonalMatrix(int size) {
        Matrix randMat = new Matrix(size, size, RAND_ARRAY.genUniformRealArray(size));
        return new RealQR().decompose(randMat).getQ();
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
        return new CMatrix(rows, cols, RAND_ARRAY.genUniformComplexArray(rows*cols));
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
        ParameterChecks.assertRank(2, shape);
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
        return new CMatrix(rows, cols, RAND_ARRAY.genUniformComplexArray(rows*cols, min, max));
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
        ParameterChecks.assertRank(2, shape);
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
        return new CMatrix(rows, cols, RAND_ARRAY.genNormalComplexArray(rows*cols));
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param shape Shape of the resulting matrix.
     * @return A matrix filled with pseudorandom complex values with magnitudes sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public CMatrix randnCMatrix(Shape shape) {
        ParameterChecks.assertRank(2, shape);
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
        return new CMatrix(rows, cols, RAND_ARRAY.genNormalComplexArray(rows*cols, mean, std));
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
        ParameterChecks.assertRank(2, shape);
        return randnCMatrix(shape.get(0), shape.get(1), mean, std);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CooCMatrix randomSparseCMatrix(int rows, int cols, double min, double max, double sparsity) {
        return randomSparseCMatrix(new Shape(rows, cols), min, max, sparsity);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have a uniform
     * distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in {@code [min, max)}.
     */
    public CooCMatrix randomSparseCMatrix(Shape shape, double min, double max, double sparsity) {
        ParameterChecks.assertInRange(sparsity, 0, 1, "sparsity");
        int numEntries = new BigDecimal(shape.totalEntries()).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();

        return randomSparseCMatrix(shape, min, max, numEntries);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero entries. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero entries int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero entries uniformly
     * distributed in {@code [min, max)}.
     */
    public CooCMatrix randomSparseCMatrix(int rows, int cols, double min, double max, int numNonZeroEntries) {
        return randomSparseCMatrix(new Shape(rows, cols), min, max, numNonZeroEntries);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero entries. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero entries int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero entries uniformly
     * distributed in {@code [min, max)}.
     */
    public CooCMatrix randomSparseCMatrix(Shape shape, double min, double max, int numNonZeroEntries) {
        ParameterChecks.assertGreaterEq(0, numNonZeroEntries);

        CNumber[] entries = RAND_ARRAY.genUniformComplexArray(numNonZeroEntries, min, max);
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numNonZeroEntries, 0, shape.get(0), 0, shape.get(1));

        return new CooCMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Gets a pseudorandom unitary matrix. From an implementation point of view, a pseudorandom complex matrix is generated
     * as if by {@link #randomCMatrix(int, int) randomCMatrix(size, size)}. Then, a {@link ComplexQR QR}
     * decomposition is computed on this pseudorandom matrix and the {@code Q} matrix from this decomposition is returned.
     * @param size Size of the unitary matrix (i.e. the number rows and columns for the square matrix).
     * @return A pseudorandom unitary matrix.
     */
    public CMatrix randomUnitaryMatrix(int size) {
        CMatrix randMat = new CMatrix(size, size, RAND_ARRAY.genUniformComplexArray(size));
        return new ComplexQR().decompose(randMat).getQ();
    }


    /**
     * Gets a pseudorandom upper triangular matrix of the specified size. The entries will be distributed according to a
     * standard normal distribution with a mean of 0 and standard deviation of 1.
     * @param size Size if the upper triangular matrix.
     * @return A pseudorandom upper triangular matrix of the specified size.
     */
    public Matrix randomTriuMatrix(int size, int min, int max) {
        double[] entries = new double[size*size];
        double maxMin = max-min;

        for(int i=0; i<size; i++) {
            int rowOffset = i*size;
            for(int j=i; j<size; j++) {
                entries[rowOffset + j] = COMPLEX_RNG.nextDouble()*maxMin + min;
            }
        }

        return new Matrix(new Shape(size, size), entries);
    }


    /**
     * Gets a pseudorandom lower triangular matrix of the specified size. The entries will be distributed according to a
     * standard normal distribution with a mean of 0 and standard deviation of 1.
     * @param size Size if the lower triangular matrix.
     * @return A pseudorandom lower triangular matrix of the specified size.
     */
    public Matrix geRandomTrilMatrix(int size, int min, int max) {
        double[] entries = new double[size*size];
        double maxMin = max-min;

        for(int i=0; i<size; i++) {
            int rowOffset = i*size;
            for(int j=0; j<=i; j++) {
                entries[rowOffset + j] = COMPLEX_RNG.nextDouble()*maxMin + min;
            }
        }

        return new Matrix(new Shape(size, size), entries);
    }
}
