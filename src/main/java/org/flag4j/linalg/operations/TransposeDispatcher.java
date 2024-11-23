/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg.operations;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field.AbstractDenseFieldMatrix;
import org.flag4j.arrays.backend.primitive.AbstractDoubleTensor;
import org.flag4j.arrays.backend.semiring.AbstractDenseSemiringTensor;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.operations.dense.DenseTranspose;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldHermitianTranspose;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldTranspose;
import org.flag4j.linalg.operations.dense.real.RealDenseTranspose;
import org.flag4j.linalg.operations.dense.ring_ops.DenseRingHermitianTranspose;
import org.flag4j.util.ErrorMessages;


/**
 * Provides a dispatch method for dynamically choosing the best matrix transpose algorithm.
 */
public final class TransposeDispatcher {

    // TODO: These thresholds need to be updated. Perform some benchmarks to get empirical values.
    // TODO: This whole class needs to be reworked.

    /**
     * Threshold for using complex blocked algorithm.
     */
    private static final int COMPLEX_BLOCKED_THRESHOLD = 5_000;
    /**
     * Threshold for using blocked hermitian algorithm
     */
    private static final int HERMITIAN_BLOCKED_THRESHOLD = 50_000;
    /**
     * Threshold for using standard transpose implementation.
     */
    private static final int STANDARD_THRESHOLD = 1500;
    /**
     * Threshold for number of elements in matrix to use concurrent implementation.
     */
    private static final int CONCURRENT_THRESHOLD = 4_250_000;


    private TransposeDispatcher() {
        // Hide default constructor.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Dispatches a matrix transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    public static Matrix dispatch(Matrix src) {

        double[] dest;

        TransposeAlgorithms algorithm = chooseAlgorithm(src.shape);

        switch(algorithm) {
            case STANDARD:
                dest = RealDenseTranspose.standardMatrix(src.entries, src.numRows, src.numCols);
                break;
            case BLOCKED:
                dest = RealDenseTranspose.blockedMatrix(src.entries, src.numRows, src.numCols);
                break;
            case CONCURRENT_STANDARD:
                dest = RealDenseTranspose.standardMatrixConcurrent(src.entries, src.numRows, src.numCols);
                break;
            default:
                dest = RealDenseTranspose.blockedMatrixConcurrent(src.entries, src.numRows, src.numCols);
                break;
        }

        return new Matrix(src.numCols, src.numRows, dest);
    }


    /**
     * Dispatches a matrix transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    public static double[] dispatch(double[] src, Shape shape) {

        double[] dest;
        int numRows = shape.get(0);
        int numCols = shape.get(1);
        TransposeAlgorithms algorithm = chooseAlgorithm(shape);

        switch(algorithm) {
            case STANDARD:
                dest = RealDenseTranspose.standardMatrix(src, numRows, numCols);
                break;
            case BLOCKED:
                dest = RealDenseTranspose.blockedMatrix(src, numRows, numCols);
                break;
            case CONCURRENT_STANDARD:
                dest = RealDenseTranspose.standardMatrixConcurrent(src, numRows, numCols);
                break;
            default:
                dest = RealDenseTranspose.blockedMatrixConcurrent(src, numRows, numCols);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    public static <T extends Field<T>> AbstractDenseFieldMatrix<?, ?, T> dispatch(AbstractDenseFieldMatrix<?, ?, T> src) {

        Field<T>[] dest;

        TransposeAlgorithms algorithm = chooseAlgorithmComplex(src.shape); // TODO: Need an updated method for this. Or at least a name change.

        switch(algorithm) {
            case STANDARD:
                dest = DenseFieldTranspose.standardMatrix(src.entries, src.numRows, src.numCols);
                break;
            case BLOCKED:
                dest = DenseFieldTranspose.blockedMatrix(src.entries, src.numRows, src.numCols);
                break;
            case CONCURRENT_STANDARD:
                dest = DenseFieldTranspose.standardMatrixConcurrent(src.entries, src.numRows, src.numCols);
                break;
            default:
                dest = DenseFieldTranspose.blockedMatrixConcurrent(src.entries, src.numRows, src.numCols);
                break;
        }

        return src.makeLikeTensor(new Shape(src.numCols, src.numRows), dest);
    }


    /**
     * Dispatches a matrix transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Matrix to transpose.
     * @param shape Shape of the matrix to transpose.
     * @param dest Array to store the transpose result in. May be {@code null}. If not {@code null}, must be at least as large as
     * {@code src}.
     * @return If {@code dest != null} a reference to the {@code dest} array will be returned. Otherwise, if {@code dest == null}
     * then a new array will be created and returned.
     */
    public static Object[] dispatch(Object[] src, Shape shape, Object[] dest) {
        if(src == dest)
            throw new IllegalArgumentException("src and dest cannot be the same array.");

        TransposeAlgorithms algorithm = chooseAlgorithmComplex(shape); // TODO: Need an updated method for this. Or at least a name change.
        if(dest == null) dest = new Object[src.length];
        final int numRows = shape.get(0);
        final int numCols = shape.get(1);

        switch(algorithm) {
            case STANDARD:
                DenseTranspose.standardMatrix(src, numRows, numCols, dest);
                break;
            case BLOCKED:
                DenseTranspose.blockedMatrix(src, numRows, numCols, dest);
                break;
            case CONCURRENT_STANDARD:
                DenseTranspose.standardMatrixConcurrent(src, numRows, numCols, dest);
                break;
            default:
                DenseTranspose.blockedMatrixConcurrent(src, numRows, numCols, dest);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix hermitian transpose (i.e. conjugate transpose) problem to the appropriate algorithm based on its shape and
     * size.
     * @param shape Shape of the matrix to transpose and conjugate.
     * @param src Entries of the matrix to transpose and conjugate.
     * @param dest Array to store the hermitian transpose result in. May be {@code null}. If not {@code null}, must be at least as
     * large as {@code src}.
     * @return If {@code dest != null} a reference to the {@code dest} array will be returned. Otherwise, if {@code dest == null}
     * then a new array will be created and returned.
     */
    public static <T extends Field<T>> Field<T>[] dispatchHermitian(Field<T>[] src, Shape shape, Field[] dest) {
        TransposeAlgorithms algorithm = chooseAlgorithmHermitian(shape);

        if(dest == null) dest = new Field[src.length];

        if(algorithm== TransposeAlgorithms.BLOCKED) {
            dest = DenseFieldHermitianTranspose.blockedMatrixHerm(src, shape.get(0), shape.get(1));
        } else {
            dest = DenseFieldHermitianTranspose.blockedMatrixConcurrentHerm(src, shape.get(0), shape.get(1));
        }

        return dest;
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Tensor to transpose.
     * @param axis1 First axis in tensor transpose.
     * @param axis2 Second axis in tensor transpose.
     * @return The result of the tensor transpose.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor.
     */
    public static <T extends AbstractDoubleTensor<T>> T dispatchTensor(
            T src, int axis1, int axis2) {
        int rank = src.getRank();
        double[] dest;

        // TODO: Implement this strategy in each dispatchTensor method.
        if(axis1 == axis2) {
            dest = src.entries.clone();
        } else if(rank == 2) {
            dest = dispatch(src.entries, src.shape); // Matrix transpose problem.
        } else {
            TransposeAlgorithms algorithm = chooseAlgorithmTensor(src.shape.get(axis1), src.shape.get(axis2));

            dest = algorithm == TransposeAlgorithms.STANDARD ?
                    RealDenseTranspose.standard(src.entries, src.shape, axis1, axis2):
                    RealDenseTranspose.standardConcurrent(src.entries, src.shape, axis1, axis2);
        }

        return src.makeLikeTensor(src.shape.swapAxes(axis1, axis2), dest);
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Tensor to transpose.
     * @param axes Permutation of axes in the tensor transpose.
     * @return The result of the tensor transpose.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor.
     */
    public static <T extends AbstractDoubleTensor<T>> T dispatchTensor(T src, int[] axes) {
        TransposeAlgorithms algorithm = chooseAlgorithmTensor(src.entries.length);

        double[] dest = algorithm == TransposeAlgorithms.STANDARD ?
                RealDenseTranspose.standard(src.entries, src.shape, axes):
                RealDenseTranspose.standardConcurrent(src.entries, src.shape, axes);

        return src.makeLikeTensor(src.shape.permuteAxes(axes), dest);
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Tensor to transpose.
     * @param axes Permutation of axes in the tensor transpose.
     * @return The result of the tensor transpose.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor.
     */
    public static double[] dispatchTensor(double[] src, Shape shape, int[] axes) {
        TransposeAlgorithms algorithm = chooseAlgorithmTensor(src.length);

        double[] dest = algorithm == TransposeAlgorithms.STANDARD ?
                RealDenseTranspose.standard(src, shape, axes):
                RealDenseTranspose.standardConcurrent(src, shape, axes);

        return dest;
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Entries of the tensor to transpose.
     * @param srcShape Shape of the tensor to transpose.
     * @param axis1 First axis in tensor transpose.
     * @param axis2 Second axis in tensor transpose.
     * @param dest Array to store the transpose result in. May be {@code null}. Must at least as large as {@code src}.
     * @return If {@code dest != null} a reference to the {@code dest} tensor. Otherwise, if {@code dest == null} a new array will be
     * constructed and returned.
     * @throws IndexOutOfBoundsException If {@code dest.length < src.length}.
     */
    public static <T> T[] dispatchTensor(T[] src, Shape srcShape, int axis1, int axis2, T[] dest) {
        if(axis1 == axis2) {
            System.arraycopy(src, 0, dest, 0, src.length);
        } else if(srcShape.getRank() == 2) {
            dispatch(src, srcShape, dest); // Delegate to matrix transpose.
        } else {
            TransposeAlgorithms algorithm = chooseAlgorithmTensor(srcShape.get(axis1), srcShape.get(axis2));

            if(algorithm == TransposeAlgorithms.STANDARD)
                DenseTranspose.standard(src, srcShape, axis1, axis2, dest);
            else
                DenseTranspose.standardConcurrent(src, srcShape, axis1, axis2, dest);
        }

        return dest;
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Entries of tensor to transpose.
     * @param axes Permutation of axes in the tensor transpose.
     * @return The result of the tensor transpose.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor.
     */
    public static <V extends Semiring<V>> AbstractDenseSemiringTensor<?, V> dispatchTensor(
            AbstractDenseSemiringTensor<?, V> src, int[] axes) {
        Semiring<V>[] dest = new Semiring[src.entries.length];
        TransposeAlgorithms algorithm = chooseAlgorithmTensor(src.entries.length);

        if(algorithm == TransposeAlgorithms.STANDARD)
            DenseTranspose.standard(src.entries, src.shape, axes, dest);
        else
            DenseTranspose.standardConcurrent(src.entries, src.shape, axes, dest);

        return src.makeLikeTensor(src.shape.permuteAxes(axes), (V[]) dest);
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Entries of tensor to transpose.
     * @param shape Shape fo the tensor to transpose.
     * @param axes Permutation of axes in the tensor transpose.
     * @param dest Array to store the transpose result in. May be {@code null}. If {@code dest != null}, both of the following must
     * be satisfied:
     * <ul>
     *     <li>{@code dest.length >= src.length}</li>
     *     <li>{@code dest != src}</li>
     * </ul>
     * @return The result of the tensor transpose.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor or if
     * {@code dest.length < src.length}.
     * @throws IllegalArgumentException If {@code src == dest}.
     */
    public static Object[] dispatchTensor(Object[] src, Shape shape, int[] axes, Object[] dest) {
        if(src == dest) throw new IllegalArgumentException("src and dest array cannot be the same array.");
        if(dest == null) dest = new Object[src.length];

        TransposeAlgorithms algorithm = chooseAlgorithmTensor(src.length);

        if(algorithm == TransposeAlgorithms.STANDARD)
            DenseTranspose.standard(src, shape, axes, dest);
        else
            DenseTranspose.standardConcurrent(src, shape, axes, dest);

        return dest;
    }


    /**
     * Dispatches a tensor Hermitian transpose problem to the appropriate algorithm based on its shape and size.
     * @param shape Shape of the tensor to transpose.
     * @param src Entries of the tensor to transpose.
     * @param axis1 First axis in tensor transpose.
     * @param axis2 Second axis in tensor transpose.
     * @param dest Array to store the transpose result in.
     */
    public static <V extends Ring<V>> void dispatchTensorHermitian(
            Shape shape, Ring<V>[] src,
            int axis1, int axis2,
            Ring<V>[] dest) {
        TransposeAlgorithms algorithm = chooseAlgorithmTensor(shape.get(axis1), shape.get(axis2));

        if (algorithm == TransposeAlgorithms.STANDARD)
            DenseRingHermitianTranspose.standardHerm(src, shape, axis1, axis2, dest);
        else
            DenseRingHermitianTranspose.standardConcurrentHerm(src, shape, axis1, axis2, dest);
    }


    /**
     * Dispatches a tensor Hermitian transpose problem to the appropriate algorithm based on its shape and size.
     * @param Shape shape of the tensor to transpose.
     * @param src Entries of tensor to transpose.
     * @param axes Permutation of axes in the tensor transpose.
     * @param dest Array to store the result of the tensor transpose in.
     *
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor.
     */
    public static <V extends Ring<V>> void dispatchTensorHermitian(
            Shape shape, Ring<V>[] src,
            int axes[],
            Ring<V>[] dest) {
        TransposeAlgorithms algorithm = chooseAlgorithmTensor(src.length);

        if (algorithm == TransposeAlgorithms.STANDARD)
            DenseRingHermitianTranspose.standardHerm(src, shape, axes, dest);
        else
            DenseRingHermitianTranspose.standardConcurrentHerm(src, shape, axes, dest);
    }


    /**
     * Chooses the appropriate algorithm for computing a tensor transpose.
     * @param length1 Length of first axis in tensor transpose.
     * @param length2 Length of second axis in tensor transpose.
     * @return
     */
    private static TransposeAlgorithms chooseAlgorithmTensor(int length1, int length2) {
        int numEntries = length1*length2; // Number of entries involved in transpose.
        return numEntries < CONCURRENT_THRESHOLD ? TransposeAlgorithms.STANDARD : TransposeAlgorithms.CONCURRENT_STANDARD;
    }



    /**
     * Chooses the appropriate algorithm for computing a tensor transpose.
     * @param numEntries Total number of entries in tensor to transpose.
     * @return The algorithm to use for the tensor transpose.
     */
    private static TransposeAlgorithms chooseAlgorithmTensor(int numEntries) {
        return numEntries < CONCURRENT_THRESHOLD ? TransposeAlgorithms.STANDARD : TransposeAlgorithms.CONCURRENT_STANDARD;
    }



    /**
     * Chooses the appropriate matrix transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static TransposeAlgorithms chooseAlgorithm(Shape shape) {
        int numEntries = shape.totalEntries().intValueExact();
        return numEntries < CONCURRENT_THRESHOLD ? TransposeAlgorithms.BLOCKED : TransposeAlgorithms.CONCURRENT_BLOCKED;
    }


    /**
     * Chooses the appropriate matrix hermitian transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static TransposeAlgorithms chooseAlgorithmHermitian(Shape shape) {
        int numEntries = shape.totalEntries().intValueExact();
        return numEntries < HERMITIAN_BLOCKED_THRESHOLD ? TransposeAlgorithms.BLOCKED : TransposeAlgorithms.CONCURRENT_BLOCKED;
    }


    /**
     * Chooses the appropriate matrix transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static TransposeAlgorithms chooseAlgorithmComplex(Shape shape) {
        TransposeAlgorithms algorithm;

        int numEntries = shape.totalEntries().intValueExact();

        if(numEntries < STANDARD_THRESHOLD) {
            // Use standard algorithm.
            algorithm = TransposeAlgorithms.STANDARD;
        } else if(numEntries < CONCURRENT_THRESHOLD) {
            // Use blocked algorithm
            algorithm = TransposeAlgorithms.BLOCKED;
        } else {
            // Use concurrent blocked implementation.
            algorithm = TransposeAlgorithms.CONCURRENT_BLOCKED;
        }

        return algorithm;
    }


    /**
     * Simple enum class containing available algorithms for computing a matrix transpose.
     */
    private enum TransposeAlgorithms {
        /**
         * Standard transpose algorithm
         */
        STANDARD,
        /**
         * Blocked transpose algorithm
         */
        BLOCKED,
        /**
         * A concurrent implementation of the standard algorithm
         */
        CONCURRENT_STANDARD,
        /**
         * A concurrent implementation of the blocked algorithm
         */
        CONCURRENT_BLOCKED
    }
}
