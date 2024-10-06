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

package org.flag4j.operations;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldMatrixBase;
import org.flag4j.arrays.backend.FieldTensorBase;
import org.flag4j.arrays.backend.PrimitiveDoubleTensorBase;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.operations.dense.field_ops.DenseFieldHermitianTranspose;
import org.flag4j.operations.dense.field_ops.DenseFieldTranspose;
import org.flag4j.operations.dense.real.RealDenseTranspose;
import org.flag4j.util.ErrorMessages;


/**
 * Provides a dispatch method for dynamically choosing the best matrix transpose algorithm.
 */
public final class TransposeDispatcher {

    // TODO: These thresholds need to be updated. Perform some benchmarks to get empirical values.

    /**
     * Threshold for using complex blocked algorithm.
     */
    private static final int COMPLEX_BLOCKED_THRESHOLD = 5_000;
    /**
     * Threshold for using blocked hermitian algorithm
     */
    private static final int HERMATION_BLOCKED_THRESHOLD = 50_000;
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
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Dispatches a matrix transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    public static Matrix dispatch(Matrix src) {

        double[] dest;

        Algorithm algorithm = chooseAlgorithm(src.shape);

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
        Algorithm algorithm = chooseAlgorithm(shape);

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
    public static <T extends Field<T>> DenseFieldMatrixBase<?, ?, ?, ?, T> dispatch(DenseFieldMatrixBase<?, ?, ?, ?, T> src) {

        Field<T>[] dest;

        Algorithm algorithm = chooseAlgorithmComplex(src.shape); // TODO: Need an updated method for this. Or at least a name change.

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
     * @return The transpose of the source matrix.
     */
    public static <T extends Field<T>> Field<T>[] dispatch(Field<T>[] src, Shape shape) {

        Field<T>[] dest;

        Algorithm algorithm = chooseAlgorithmComplex(shape); // TODO: Need an updated method for this. Or at least a name change.
        final int numRows = shape.get(0);
        final int numCols = shape.get(1);

        switch(algorithm) {
            case STANDARD:
                dest = DenseFieldTranspose.standardMatrix(src, numRows, numCols);
                break;
            case BLOCKED:
                dest = DenseFieldTranspose.blockedMatrix(src, numRows, numCols);
                break;
            case CONCURRENT_STANDARD:
                dest = DenseFieldTranspose.standardMatrixConcurrent(src, numRows, numCols);
                break;
            default:
                dest = DenseFieldTranspose.blockedMatrixConcurrent(src, numRows, numCols);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix transpose problem to the appropriate algorithm based in its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    @Deprecated
    public static CMatrix dispatch(CMatrix src) {
        Field<Complex128>[] dest;
        Algorithm algorithm = chooseAlgorithm(src.shape);

        if(algorithm == Algorithm.BLOCKED)
            dest = DenseFieldTranspose.blockedMatrix(src.entries, src.numRows, src.numCols);
        else
            dest = DenseFieldTranspose.blockedMatrixConcurrent(src.entries, src.numRows, src.numCols);

        return new CMatrix(src.numCols, src.numRows, dest);
    }


    /**
     * Dispatches a matrix hermitian transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    @Deprecated
    public static CMatrix dispatchHermitian(CMatrix src) {
        Field<Complex128>[] dest;

        Algorithm algorithm = chooseAlgorithmHermitian(src.shape);

        if(algorithm==Algorithm.BLOCKED) {
            dest = DenseFieldTranspose.blockedMatrixHerm(src.entries, src.numRows, src.numCols);
        } else {
            dest = DenseFieldTranspose.blockedMatrixConcurrentHerm(src.entries, src.numRows, src.numCols);
        }

        return new CMatrix(src.numCols, src.numRows, dest);
    }


    /**
     * Dispatches a matrix hermitian transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    public static <T extends Field<T>> DenseFieldMatrixBase<?, ?, ?, ?, T> dispatchHermitian(DenseFieldMatrixBase<?, ?, ?, ?, T> src) {
        Field<T>[] dest;

        Algorithm algorithm = chooseAlgorithmHermitian(src.shape);

        if(algorithm==Algorithm.BLOCKED) {
            dest = DenseFieldHermitianTranspose.blockedMatrixHerm(src.entries, src.numRows, src.numCols);
        } else {
            dest = DenseFieldHermitianTranspose.blockedMatrixConcurrentHerm(src.entries, src.numRows, src.numCols);
        }

        return src.makeLikeTensor(new Shape(src.numCols, src.numRows), dest);
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Tensor to transpose.
     * @param axis1 First axis in tensor transpose.
     * @param axis2 Second axis in tensor transpose.
     * @return The result of the tensor transpose.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor.
     */
    public static <T extends PrimitiveDoubleTensorBase<T, T>> T dispatchTensor(PrimitiveDoubleTensorBase<T, T> src,
                                                                               int axis1, int axis2) {
        int rank = src.getRank();
        double[] dest;

        // TODO: Implement this strategy in each dispatchTensor method.
        if(axis1 == axis2) {
            dest = src.entries.clone();
        } else if(rank == 2) {
            dest = dispatch(src.entries, src.shape); // Matrix transpose problem.
        } else {
            Algorithm algorithm = chooseAlgorithmTensor(src.shape.get(axis1), src.shape.get(axis2));

            dest = algorithm == Algorithm.STANDARD ?
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
    public static <T extends PrimitiveDoubleTensorBase<T, T>> T dispatchTensor(PrimitiveDoubleTensorBase<T, T> src, int[] axes) {
        double[] dest;
        Algorithm algorithm = chooseAlgorithmTensor(src.entries.length);

        dest = algorithm == Algorithm.STANDARD ?
                RealDenseTranspose.standard(src.entries, src.shape, axes):
                RealDenseTranspose.standardConcurrent(src.entries, src.shape, axes);

        return src.makeLikeTensor(src.shape.swapAxes(axes), dest);
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Tensor to transpose.
     * @param axis1 First axis in tensor transpose.
     * @param axis2 Second axis in tensor transpose.
     * @return The result of the tesnsor transpose.
     */
    public static <T extends FieldTensorBase<T, T, V>, V extends Field<V>> T dispatchTensor(
            FieldTensorBase<T, T, V> src, int axis1, int axis2) {
        Field<V>[] dest;

        if(axis1 == axis2) {
            dest = src.entries.clone();
        } else if(src.getRank() == 2) {
            dest = dispatch(src.entries, src.shape); // Delegate to matrix transpose.
        } else {
            Algorithm algorithm = chooseAlgorithmTensor(src.shape.get(axis1), src.shape.get(axis2));
            dest = algorithm == Algorithm.STANDARD ?
                    DenseFieldTranspose.standard(src.entries, src.shape, axis1, axis2):
                    DenseFieldTranspose.standardConcurrent(src.entries, src.shape, axis1, axis2);
        }

        return src.makeLikeTensor(src.shape.swapAxes(axis1, axis2), (V[]) dest);
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Entries of tensor to transpose.
     * @param axes Permutation of axes in the tensor transpose.
     * @return The result of the tensor transpose.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor.
     */
    public static <T extends FieldTensorBase<T, T, V>, V extends Field<V>> T dispatchTensor(
            FieldTensorBase<T, T, V> src, int[] axes) {
        Field<V>[] dest;
        Algorithm algorithm = chooseAlgorithmTensor(src.entries.length);

        dest = algorithm == Algorithm.STANDARD ?
                DenseFieldTranspose.standard(src.entries, src.shape, axes):
                DenseFieldTranspose.standardConcurrent(src.entries, src.shape, axes);

        return src.makeLikeTensor(src.shape.swapAxes(axes), (V[]) dest);
    }


    /**
     * Dispatches a tensor Hermitian transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Tensor to transpose.
     * @param axis1 First axis in tensor transpose.
     * @param axis2 Second axis in tensor transpose.
     * @return
     */
    public static <T extends FieldTensorBase<T, T, V>, V extends Field<V>> T dispatchTensorHermitian(
            FieldTensorBase<T, T, V> src,
            int axis1,
            int axis2) {
        Field<V>[] dest;
        Algorithm algorithm = chooseAlgorithmTensor(src.shape.get(axis1), src.shape.get(axis2));

        dest = algorithm == Algorithm.STANDARD ?
                DenseFieldHermitianTranspose.standardHerm(src.entries, src.shape, axis1, axis2):
                DenseFieldHermitianTranspose.standardConcurrentHerm(src.entries, src.shape, axis1, axis2);

        return src.makeLikeTensor(src.shape.swapAxes(axis1, axis2), (V[]) dest);
    }


    /**
     * Dispatches a tensor Hermitian transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Entries of tensor to transpose.
     * @param axes Permutation of axes in the tensor transpose.
     * @return The result of the tensor transpose.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within the {@code src} tensor.
     */
    public static <T extends FieldTensorBase<T, T, V>, V extends Field<V>> T dispatchTensorHermitian(
            FieldTensorBase<T, T, V> src,
            int[] axes) {
        Field<V>[] dest;
        Algorithm algorithm = chooseAlgorithmTensor(src.entries.length);

        dest = algorithm == Algorithm.STANDARD ?
                DenseFieldHermitianTranspose.standardHerm(src.entries, src.shape, axes):
                DenseFieldHermitianTranspose.standardConcurrentHerm(src.entries, src.shape, axes);

        return src.makeLikeTensor(src.shape.swapAxes(axes), (V[]) dest);
    }


    /**
     * Chooses the appropriate algorithm for computing a tensor transpose.
     * @param length1 Length of first axis in tensor transpose.
     * @param length2 Length of second axis in tensor transpose.
     * @return
     */
    private static Algorithm chooseAlgorithmTensor(int length1, int length2) {
        int numEntries = length1*length2; // Number of entries involved in transpose.
        return numEntries < CONCURRENT_THRESHOLD ? Algorithm.STANDARD : Algorithm.CONCURRENT_STANDARD;
    }



    /**
     * Chooses the appropriate algorithm for computing a tensor transpose.
     * @param numEntries Total number of entries in tensor to transpose.
     * @return The algorithm to use for the tensor transpose.
     */
    private static Algorithm chooseAlgorithmTensor(int numEntries) {
        return numEntries < CONCURRENT_THRESHOLD ? Algorithm.STANDARD : Algorithm.CONCURRENT_STANDARD;
    }



    /**
     * Chooses the appropriate matrix transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static Algorithm chooseAlgorithm(Shape shape) {
        int numEntries = shape.totalEntries().intValueExact();
        return numEntries < CONCURRENT_THRESHOLD ? Algorithm.BLOCKED : Algorithm.CONCURRENT_BLOCKED;
    }


    /**
     * Chooses the appropriate matrix hermitian transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static Algorithm chooseAlgorithmHermitian(Shape shape) {
        int numEntries = shape.totalEntries().intValueExact();
        return numEntries < HERMATION_BLOCKED_THRESHOLD ? Algorithm.BLOCKED : Algorithm.CONCURRENT_BLOCKED;
    }


    /**
     * Chooses the appropriate matrix transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static Algorithm chooseAlgorithmComplex(Shape shape) {
        Algorithm algorithm;

        int numEntries = shape.totalEntries().intValueExact();

        if(numEntries < STANDARD_THRESHOLD) {
            // Use standard algorithm.
            algorithm = Algorithm.STANDARD;
        } else if(numEntries < CONCURRENT_THRESHOLD) {
            // Use blocked algorithm
            algorithm = Algorithm.BLOCKED;
        } else {
            // Use concurrent blocked implementation.
            algorithm = Algorithm.CONCURRENT_BLOCKED;
        }

        return algorithm;
    }


    /**
     * Simple enum class containing available algorithms for computing a matrix transpose.
     */
    private enum Algorithm {
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
