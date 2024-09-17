package org.flag4j.rng;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.*;
import org.flag4j.linalg.decompositions.qr.ComplexQR;
import org.flag4j.linalg.decompositions.qr.RealQR;
import org.flag4j.util.ParameterChecks;


/**
 * <p>An instance of this class is used for generating streams of pseudorandom dense tensors, matrices, and vectors.</p>
 *
 * <p>The random values in instances of this class are generated by an instance of the {@link RandomArray}.</p>
 *
 * @see RandomSparseTensor
 */
public class RandomDenseTensor {

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
    public RandomDenseTensor() {
        COMPLEX_RNG = new RandomComplex();
        RAND_ARRAY = new RandomArray(COMPLEX_RNG);
    }
    

    /**
     * Constructs a pseudorandom tensor generator with a specified seed. Use this constructor for reproducible results.
     * @param seed Seed of the pseudorandom tensor generator.
     */
    public RandomDenseTensor(long seed) {
        COMPLEX_RNG = new RandomComplex(seed);
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
        return new CTensor(shape, RAND_ARRAY.genUniformComplex128Array(shape.totalEntries().intValueExact()));
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
        return new CTensor(shape, RAND_ARRAY.genUniformComplex128Array(
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
        return new CTensor(shape, RAND_ARRAY.genNormalComplex128Array(shape.totalEntries().intValueExact()));
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
        return new CTensor(shape, RAND_ARRAY.genNormalComplex128Array(
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
        return new CVector(RAND_ARRAY.genUniformComplex128Array(size));
    }



    /**
     * Generates a vector filled with pseudorandom values with magnitudes uniformly distributed in {@code [min, max)}.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values with magnitudes uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code min} is negative or if {@code max} is less than {@code min}.
     */
    public CVector randomCVector(int size, double min, double max) {
        return new CVector(RAND_ARRAY.genUniformComplex128Array(size, min, max));
    }


    /**
     * Generates a vector filled with pseudorandom values with magnitudes sampled from a normal distribution with a
     * mean of 0.0 and a standard deviation of 1.0.
     * @param size Size of the vector to generate.
     * @return A vector filled with pseudorandom values with magnitudes sampled from a normal distribution with a
     * mean of 0.0 and a standard deviation of 1.0.
     */
    public CVector randnCVector(int size) {
        return new CVector(RAND_ARRAY.genNormalComplex128Array(size));
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
        return new CVector(RAND_ARRAY.genNormalComplex128Array(size, mean, std));
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
        ParameterChecks.ensureRank(shape, 2);
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
        ParameterChecks.ensureRank(shape, 2);
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
        ParameterChecks.ensureRank(shape, 2);
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
        ParameterChecks.ensureRank(shape, 2);
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
        return new CMatrix(rows, cols, RAND_ARRAY.genUniformComplex128Array(rows*cols));
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
        ParameterChecks.ensureRank(shape, 2);
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
        return new CMatrix(rows, cols, RAND_ARRAY.genUniformComplex128Array(rows*cols, min, max));
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
        ParameterChecks.ensureRank(shape, 2);
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
        return new CMatrix(rows, cols, RAND_ARRAY.genNormalComplex128Array(rows*cols));
    }


    /**
     * Generates a matrix filled with pseudorandom complex values with magnitudes sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     * @param shape Shape of the resulting matrix.
     * @return A matrix filled with pseudorandom complex values with magnitudes sampled from a normal distribution with a mean of 0.0 and
     * a standard deviation of 1.0.
     */
    public CMatrix randnCMatrix(Shape shape) {
        ParameterChecks.ensureRank(shape, 2);
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
        return new CMatrix(rows, cols, RAND_ARRAY.genNormalComplex128Array(rows*cols, mean, std));
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
        ParameterChecks.ensureRank(shape, 2);
        return randnCMatrix(shape.get(0), shape.get(1), mean, std);
    }


    /**
     * Gets a pseudorandom unitary matrix. From an implementation point of view, a pseudorandom complex matrix is generated
     * as if by {@link #randomCMatrix(int, int) randomCMatrix(size, size)}. Then, a {@link ComplexQR QR}
     * decomposition is computed on this pseudorandom matrix and the {@code Q} matrix from this decomposition is returned.
     * @param size Size of the unitary matrix (i.e. the number rows and columns for the square matrix).
     * @return A pseudorandom unitary matrix.
     */
    public CMatrix randomUnitaryMatrix(int size) {
        CMatrix randMat = new CMatrix(size, size, RAND_ARRAY.genUniformComplex128Array(size));
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