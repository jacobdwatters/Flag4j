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

package org.flag4j.linalg;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.common.ring_ops.CompareRing;
import org.flag4j.linalg.ops.sparse.coo.real.RealSparseNorms;
import org.flag4j.linalg.ops.sparse.coo.ring_ops.CooRingNorms;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.function.Function;

/**
 * Utility class containing static methods for computing norms of matrices.
 */
public final class MatrixNorms {
    // TODO: Add nuclear norms.

    private MatrixNorms() {
        // Hide default constructor for utility class
    }


    /**
     * <p>Computes the L<sub>2, 2</sub> (Frobenius) norm of a real dense matrix.
     *
     * <p>This is equivalent to {@link #norm(Matrix, double, double) norm(src, 2, 2)}.
     * This will be equal to the largest singular value of the matrix. However, this method should generally be preferred over
     * {@link #norm(Matrix, double, double)} as it <i>may</i> be slightly more efficient.
     *
     * @param src Matrix to compute the L<sub>2, 2</sub> norm of.
     *
     * @return the L<sub>2, 2</sub> of this tensor.
     */
    public static double norm(Matrix src) {
        return VectorNorms.norm(src.data);
    }


    /**
     * Computes the p-norm of this tensor. Equivalent to calling {@link #norm(Matrix, double, double) norm(p, p)}
     *
     * @param src Matrix to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(Matrix src, double p) {
        if(p == Double.POSITIVE_INFINITY) {
            return matrixInfNorm(src.shape, src.data, RealProperties::max);
        } else if(p == Double.NEGATIVE_INFINITY) {
            return matrixInfNorm(src.shape, src.data, RealProperties::min);
        } else if (p == 1) {
            return matrixL1Norm(src.shape, src.data, RealProperties::max);
        } else if (p == -1) {
            return matrixL1Norm(src.shape, src.data, RealProperties::min);
        } else if (p == 2) {
            return svdBasedNorm(src, RealProperties::max);
        } else if (p == -2) {
            return svdBasedNorm(src, RealProperties::min);
        } else {
            throw new LinearAlgebraException("Unsupported norm type: p = " + p
                    + ". Supported values are: 1, -1, 2, -2, Double.POSITIVE_INFINITY, and Double.NEGATIVE_INFINITY.");
        }
    }


    /**
     * Computes the maximum norm of this matrix. That is, the maximum value in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The maximum norm of this matrix.
     * @see #infNorm(Matrix)
     */
    public static double maxNorm(Matrix src) {
        return RealProperties.maxAbs(src.data);
    }


    /**
     * Computes the infinite norm of this matrix. that is the maximum row sum in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The infinite norm of this matrix.
     * @see #maxNorm(Matrix)
     */
    public static double infNorm(Matrix src) {
        return matrixInfNorm(src.shape, src.data, RealProperties::max);
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(Matrix src, double p, double q) {
        return matrixNormLpq(src.data, src.shape, p, q);
    }


    /**
     * <p>Computes the L<sub>2, 2</sub> (Frobenius) norm of a real dense matrix.
     *
     * <p>This is equivalent to {@link #norm(Matrix, double, double) norm(src, 2, 2)}.
     * This will be equal to the largest singular value of the matrix. However, this method should generally be preferred over
     * {@link #norm(Matrix, double, double)} as it <i>may</i> be slightly more efficient.
     *
     * @param src Matrix to compute the L<sub>2, 2</sub> norm of.
     *
     * @return the L<sub>2, 2</sub> of this tensor.
     */
    public static double norm(CMatrix src) {
        return VectorNorms.norm(src.data);
    }


    /**
     * Computes the p-norm of this tensor. Equivalent to calling {@link #norm(Matrix, double, double) norm(p, p)}
     *
     * @param src Matrix to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CMatrix src, double p) {
        if(p == Double.POSITIVE_INFINITY) {
            return matrixInfNorm(src.shape, src.data, RealProperties::max);
        } else if(p == Double.NEGATIVE_INFINITY) {
            return matrixInfNorm(src.shape, src.data, RealProperties::min);
        } else if (p == 1) {
            return matrixL1Norm(src.shape, src.data, RealProperties::max);
        } else if (p == -1) {
            return matrixL1Norm(src.shape, src.data, RealProperties::min);
        } else if (p == 2) {
            return svdBasedNorm(src, RealProperties::max);
        } else if (p == -2) {
            return svdBasedNorm(src, RealProperties::min);
        } else {
            throw new LinearAlgebraException("Unsupported norm type: p = " + p
                    + ". Supported values are: 1, -1, 2, -2, Double.POSITIVE_INFINITY, and Double.NEGATIVE_INFINITY.");
        }
    }


    /**
     * Computes the maximum norm of this matrix. That is, the maximum value in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The maximum norm of this matrix.
     * @see #infNorm(Matrix)
     */
    public static double maxNorm(CMatrix src) {
        return CompareRing.maxAbs(src.data);
    }


    /**
     * Computes the infinite norm of this matrix. that is the maximum row sum in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The infinite norm of this matrix.
     * @see #maxNorm(Matrix)
     */
    public static double infNorm(CMatrix src) {
        return matrixInfNorm(src.shape, src.data, RealProperties::max);
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CMatrix src, double p, double q) {
        return matrixNormLpq(src.data, src.shape, p, q);
    }


    // ------------------------------ Sparse COO Matrices ------------------------------
    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src Matrix to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CooMatrix src, double p, double q) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? RealSparseNorms.matrixNormLpq(src, p, q) :
                norm(src.toDense(), p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CooMatrix src) {
        return RealProperties.maxAbs(src.data);
    }


    /**
     * Computes the L<sub>2, 2</sub> (Frobenius) norm of this complex COO matrix. This is equivalent to
     * {@link #norm(CooMatrix, double, double) norm(src, 2, 2)}.
     *
     * @param src Matrix to compute the L<sub>2, 2</sub> norm of.
     * @return the L<sub>2, 2</sub> (Frobenius) norm of this tensor.
     */
    public static double norm(CooMatrix src) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? RealSparseNorms.matrixNormL2(src) :
                norm(src.toDense());
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src Matrix to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CooCMatrix src, double p, double q) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? CooRingNorms.matrixNormLpq(src, p, q) :
                norm(src.toDense(), p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CooCMatrix src) {
        return matrixMaxNorm(src.data);
    }


    /**
     * Computes the L<sub>2, 2</sub> (Frobenius) norm of this complex COO matrix. This is equivalent to
     * {@link #norm(CooCMatrix, double, double) norm(src, 2, 2)}.
     *
     * @param src Matrix to compute the L<sub>2, 2</sub> norm of.
     * @return the L<sub>2, 2</sub> (Frobenius) norm of this tensor.
     */
    public static double norm(CooCMatrix src) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? CooRingNorms.matrixNormL22(src) :
                norm(src.toDense());
    }

    // ------------------------------ Sparse CSR Matrices ------------------------------

    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src Matrix to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CsrMatrix src, double p, double q) {
        double norm = 0;
        double pOverQ = p / q;

        // stores intermediate column norms.
        double[] colNorms = new double[src.numCols];

        // Accumulate column-wise norms.
        for (int row = 0; row < src.numRows; row++) {
            int start = src.rowPointers[row];
            int end = src.rowPointers[row + 1];

            for (int idx = start; idx < end; idx++) {
                int col = src.colIndices[idx];
                double value = src.data[idx];

                colNorms[col] += Math.pow(Math.abs(value), p);
            }
        }

        // Compute the q-norm of the column norms.
        for (double colNorm : colNorms)
            if (colNorm > 0) norm += Math.pow(colNorm, pOverQ);

        return Math.pow(norm, 1.0 / q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CsrMatrix src) {
        return RealProperties.maxAbs(src.data);
    }


    /**
     * Computes the Frobenius of this matrix. This is equivalent to {@link #norm(CsrMatrix, double, double) norm(src, 2, 2)}.
     *
     * @param src Matrix to compute the norm of.
     * @return the Frobenius of this matrix.
     */
    public double norm(CsrMatrix src) {
        return VectorNorms.norm(src.data); // Zeros do not contribute to this norm.
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src Matrix to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CsrCMatrix src, double p, double q) {
        double norm = 0;
        double pOverQ = p / q;

        // stores intermediate column norms.
        double[] colNorms = new double[src.numCols];

        // Accumulate column-wise norms.
        for (int row = 0; row < src.numRows; row++) {
            int start = src.rowPointers[row];
            int end = src.rowPointers[row + 1];

            for (int idx = start; idx < end; idx++) {
                int col = src.colIndices[idx];
                double value = src.data[idx].mag();

                colNorms[col] += Math.pow(value, p);
            }
        }

        // Compute the q-norm of the column norms.
        for (double colNorm : colNorms)
            if (colNorm > 0) norm += Math.pow(colNorm, pOverQ);

        return Math.pow(norm, 1.0 / q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CsrCMatrix src) {
        return CompareRing.maxAbs(src.data);
    }


    /**
     * Computes the Frobenius of this matrix. This is equivalent to {@link #norm(CsrCMatrix, double, double) norm(src, 2, 2)}.
     *
     * @param src Matrix to compute the norm of.
     * @return the Frobenius of this matrix.
     */
    public double norm(CsrCMatrix src) {
        return VectorNorms.norm(src.data); // Zeros do not contribute to this norm.
    }


    // -------------------------------------------------- Low-level implementations --------------------------------------------------

    /**
     * Compute the L<sub>p, q</sub> norm of a matrix.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p First parameter in L<sub>p, q</sub> norm.
     * @param q Second parameter in L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of the matrix.
     * @throws IllegalArgumentException If {@code p} or {@code q} is less than 1.
     */
    private static double matrixNormLpq(double[] src, Shape shape, double p, double q) {
        if (p < 1 || q < 1) {
            throw new LinearAlgebraException(String.format(
                    "Invalid Norm. Expecting p and q to be greater than or equal to 1" +
                            " but got p=%f, q=%f.", p, q));
        }

        double norm = 0;
        double colSum;
        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int j=0; j<cols; j++) {
            colSum=0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(Math.abs(src[i*cols + j]), p);
            }
            norm += Math.pow(colSum, q/p);
        }

        return Math.pow(norm, 1.0/q);
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    private static double matrixMaxNorm(double[] src) {
        return RealProperties.maxAbs(src);
    }


    /**
     * Computes the L<sub>1</sub> norm of a matrix.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param op Operation to apply to row sums.
     *
     * @return The L<sub>1</sub> norm of the matrix.
     */
    private static double matrixInfNorm(Shape shape, double[] src, Function<double[], Double> op) {
        // TODO: Update Javadoc (also for public methods).
        //  Since `op` is provided, this may need a more general name like rowInducedNorm(...)
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] rowSums = new double[rows];

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;
            for(int j=0; j<cols; j++)
                rowSums[i] += Math.abs(src[rowOffset + j]);
        }

        return op.apply(rowSums);
    }


    /**
     * Computes the L<sub>2</sub> (spectral) norm of a matrix.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param op Operation to apply to column sums.
     *
     * @return The L<sub>2</sub> norm of the matrix.
     */
    private static double svdBasedNorm(Matrix src, Function<double[], Double> op) {
        // TODO: Update javadoc (also for public methods).
        Matrix sigmas = new RealSVD(false).decompose(src).getS();
        return op.apply(sigmas.getDiag().data);
    }


    /**
     * Computes the L2 norm of a matrix. That is, the maximum value in this matrix.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param op Operation to apply to column sums.
     *
     * @return The infinity norm of the matrix.
     */
    private static double matrixL1Norm(Shape shape, double[] src, Function<double[], Double> op) {
        // TODO: Update javadoc (also for public methods).
        //   Since `op` is provided, this may need a more general name like columInducedNorm(...)
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] colSums = new double[cols];

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;
            for(int j=0; j<cols; j++)
                colSums[j] += Math.abs(src[rowOffset + j]);
        }

        return op.apply(colSums);
    }


    /**
     * Compute the L<sub>p, q</sub> norm of a matrix.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p First parameter in L<sub>p, q</sub> norm.
     * @param q Second parameter in L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of the matrix.
     * @throws IllegalArgumentException If {@code p} or {@code q} is less than 1.
     */
    public static double matrixNormLpq(Complex128[] src, Shape shape, double p, double q) {
        ValidateParameters.ensureGreaterEq(1, p, q);

        double norm = 0;
        double colSum;
        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int j=0; j<cols; j++) {
            colSum=0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(src[i*cols + j].mag(), p);
            }
            norm += Math.pow(colSum, q/p);
        }

        return Math.pow(norm, 1.0/q);
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    private static double matrixMaxNorm(Complex128[] src) {
        return CompareRing.maxAbs(src);
    }


    /**
     * Computes the L<sub>1</sub> norm of a matrix.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param op Operation to apply to row sums.
     *
     * @return The L<sub>1</sub> norm of the matrix.
     */
    private static double matrixInfNorm(Shape shape, Complex128[] src, Function<double[], Double> op) {
        // TODO: Update Javadoc (also for public methods).
        //  Since `op` is provided, this may need a more general name like rowInducedNorm(...)
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] rowSums = new double[rows];

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;
            for(int j=0; j<cols; j++)
                rowSums[i] += src[rowOffset + j].mag();
        }

        return op.apply(rowSums);
    }


    /**
     * Computes the L<sub>2</sub> (spectral) norm of a matrix.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param op Operation to apply to column sums.
     *
     * @return The L<sub>2</sub> norm of the matrix.
     */
    private static double svdBasedNorm(CMatrix src, Function<double[], Double> op) {
        // TODO: Update javadoc (also for public methods).
        Matrix sigmas = new ComplexSVD(false).decompose(src).getS();
        return op.apply(sigmas.getDiag().data);
    }


    /**
     * Computes the L2 norm of a matrix. That is, the maximum value in this matrix.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     * @param op Operation to apply to column sums.
     *
     * @return The infinity norm of the matrix.
     */
    private static double matrixL1Norm(Shape shape, Complex128[] src, Function<double[], Double> op) {
        // TODO: Update javadoc (also for public methods).
        //   Since `op` is provided, this may need a more general name like columInducedNorm(...)
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] colSums = new double[cols];

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;
            for(int j=0; j<cols; j++)
                colSums[j] += src[rowOffset + j].mag();
        }

        return op.apply(colSums);
    }
}
