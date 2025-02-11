/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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


import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.rng.distributions.Complex128UniformDisk;
import org.flag4j.rng.distributions.RealUniform;
import org.flag4j.util.ArrayJoiner;
import org.flag4j.util.ValidateParameters;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * A utility class for generating random sparse tensors and matrices with customizable sparsity and distributions.
 *
 * <h2>Features:</h2>
 * <ul>
 *   <li>Generate sparse matrices and tensors with specified sparsity levels.</li>
 *   <li>Support for uniform and annular (complex valued) distributions for non-zero values.</li>
 *   <li>Create symmetric sparse matrices.</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 *     RandomSparseTensor generator = new RandomSparseTensor(12345L);
 *     CooMatrix sparseMatrix = generator.randomCooMatrix(100, 100, 0, 10, 0.95);
 *     CsrMatrix csrMatrix = generator.randomCsrMatrix(new Shape(50, 50), 0, 1, 0.9);
 * }</pre>
 *
 * @see RandomComplex
 * @see RandomDenseTensor
 * @see RandomArray
 */
public class RandomSparseTensor {


    /**
     * Complex pseudorandom number generator.
     */
    private final RandomComplex COMPLEX_RNG;
    /**
     * Generator for random arrays.
     */
    private final RandomArray RAND_ARRAY;


    /**
     * Constructs a new pseudorandom tensor generator with a seed which is unlikely to be the same as other
     * from any other invocation of this constructor.
     */
    public RandomSparseTensor() {
        COMPLEX_RNG = new RandomComplex();
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
    }


    /**
     * Constructs a pseudorandom tensor generator with a specified seed. Use this constructor for reproducible results.
     * @param seed Seed of the pseudorandom tensor generator.
     */
    public RandomSparseTensor(long seed) {
        COMPLEX_RNG = new RandomComplex(seed);
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
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
        ValidateParameters.ensureRank(shape, 2);

        int numEntries = new BigDecimal(shape.totalEntries()).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();

        return randomCooMatrix(shape, min, max, numEntries);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(int rows, int cols, double min, double max, int numNonZeroEntries) {
        return randomCooMatrix(new Shape(rows, cols), min, max, numNonZeroEntries);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in {@code [min, max)}.
     */
    public CooMatrix randomCooMatrix(Shape shape, double min, double max, int numNonZeroEntries) {
        ValidateParameters.ensureGreaterEq(0, numNonZeroEntries);
        ValidateParameters.ensureLessEq(shape.totalEntries(), numNonZeroEntries, "numNonZeroEntries");

        double[] data = new double[numNonZeroEntries];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numNonZeroEntries, 0, shape.get(0), 0, shape.get(1));

        return new CooMatrix(shape, data, indices[0], indices[1]);
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
        ValidateParameters.ensureRank(shape, 2);

        int numEntries = new BigDecimal(shape.totalEntries()).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();

        return randomCooMatrix(shape, min, max, numEntries).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(int rows, int cols, double min, double max, int numNonZeroEntries) {
        return randomCooMatrix(new Shape(rows, cols), min, max, numNonZeroEntries).toCsr();
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in {@code [min, max)}. Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Minimum value for random non-zero values in the sparse matrix.
     * @param max Maximum value for random non-zero values
     * @param numNonZeroEntries Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in {@code [min, max)}.
     */
    public CsrMatrix randomCsrMatrix(Shape shape, double min, double max, int numNonZeroEntries) {
        ValidateParameters.ensureGreaterEq(0, numNonZeroEntries);
        ValidateParameters.ensureLessEq(shape.totalEntries(), numNonZeroEntries, "numNonZeroEntries");

        double[] data = new double[numNonZeroEntries];
        RandomArray.randomFill(data, new RealUniform(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numNonZeroEntries, 0, shape.get(0), 0, shape.get(1));

        return new CooMatrix(shape, data, indices[0], indices[1]).toCsr();
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
        ValidateParameters.ensureInRange(sparsity, 0, 1, "sparsity");
        Shape shape = new Shape(size, size);

        int numEntries = new BigDecimal(size).pow(2).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();
        numEntries /= 2;

        // Generate half of the random data.
        double[] entries = new double[numEntries];
        RandomArray.randomFill(entries, new RealUniform(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numEntries, 0, shape.get(0), 0, shape.get(1));

        // Mirror data across diagonal.
        entries = ArrayJoiner.join(entries, entries);
        indices = new int[][]{
                ArrayJoiner.join(indices[0], indices[1]),
                ArrayJoiner.join(indices[1], indices[0])
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
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have
     * a uniform distribution in the annulus (i.e. washer) with inner radius {@code min} (inclusive) and outer radius
     * {@code max} (exclusive). Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param min Inner radius of the annular distribution (inclusive).
     * @param max Outer radius of the annular distribution (exclusive).
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in an annulus.
     */
    public CooCMatrix randomSparseCMatrix(int rows, int cols, double min, double max, double sparsity) {
        return randomSparseCMatrix(new Shape(rows, cols), min, max, sparsity);
    }


    /**
     * Generates a random sparse matrix with the specified sparsity. The non-zero values will have
     * a uniform distribution in the annulus (i.e. washer) with inner radius {@code min} (inclusive) and outer radius
     * {@code max} (exclusive). Values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Inner radius of the annular distribution (inclusive).
     * @param max Outer radius of the annular distribution (exclusive).
     * @param sparsity Desired sparsity of the resulting matrix. i.e. the percent of values which are zero. Must be
     *                 a value in {@code [0.0, 1.0]}.
     * @return A sparse matrix with sparsity approximately equal to {@code sparsity} filled with random values uniformly
     * distributed in an annulus.
     */
    public CooCMatrix randomSparseCMatrix(Shape shape, double min, double max, double sparsity) {
        ValidateParameters.ensureInRange(sparsity, 0, 1, "sparsity");
        int numEntries = new BigDecimal(shape.totalEntries()).multiply(BigDecimal.valueOf(1.0-sparsity))
                .setScale(0, RoundingMode.HALF_UP).intValueExact();

        return randomSparseCMatrix(shape, min, max, numEntries);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in the annulus (i.e. washer) with inner radius {@code min} (inclusive) and outer radius
     * {@code max} (exclusive). Values will be uniformly distributed throughout the matrix.
     * @param rows Number of rows in the random sparse matrix.
     * @param cols Number of columns in the random sparse matrix.
     * @param min Inner radius of the annular distribution (inclusive).
     * @param max Outer radius of the annular distribution (exclusive).
     * @param numNonZeroEntries Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in an annulus.
     */
    public CooCMatrix randomSparseCMatrix(int rows, int cols, double min, double max, int numNonZeroEntries) {
        return randomSparseCMatrix(new Shape(rows, cols), min, max, numNonZeroEntries);
    }


    /**
     * Generates a random sparse matrix with the specified number of non-zero data. The non-zero values will have
     * a uniform distribution in the annulus (i.e. washer) with inner radius {@code min} (inclusive) and outer radius
     * {@code max} (exclusive). Non-zero values will be uniformly distributed throughout the matrix.
     * @param shape Shape of the sparse matrix to generate.
     * @param min Inner radius of the annular distribution (inclusive).
     * @param max Outer radius of the annular distribution (exclusive).
     * @param numNonZeroEntries Desired number of non-zero data int the random sparse matrix.
     * @return A sparse matrix filled with the specified number of non-zero data uniformly
     * distributed in an annulus.
     */
    public CooCMatrix randomSparseCMatrix(Shape shape, double min, double max, int numNonZeroEntries) {
        ValidateParameters.ensureGreaterEq(0, numNonZeroEntries);

        Complex128[] entries = new Complex128[numNonZeroEntries];
        RandomArray.randomFill(entries, new Complex128UniformDisk(COMPLEX_RNG, min, max));
        int[][] indices = RAND_ARRAY.randomUniqueIndices2D(numNonZeroEntries, 0, shape.get(0), 0, shape.get(1));

        return new CooCMatrix(shape, entries, indices[0], indices[1]);
    }
}
