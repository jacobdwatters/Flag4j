/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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


import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.arrays_old.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.operations_old.common.complex.AggregateComplex;
import org.flag4j.operations_old.common.real.AggregateReal;
import org.flag4j.operations_old.sparse.coo.complex.ComplexSparseNorms;
import org.flag4j.operations_old.sparse.coo.real.RealSparseNorms;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * Utility class for computing norms of matrices.
 */
public class MatrixNorms {

    private MatrixNorms() {
        // Hide default constructor for utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(MatrixOld, double) norm(2)}.
     * This will be equal to the largest singular value of the matrix.
     *
     * @param src MatrixOld to compute norm of.
     *
     * @return the 2-norm of this tensor.
     */
    public static double norm(MatrixOld src) {
        return TensorNorms.tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor. Equivalent to calling {@link #norm(MatrixOld, double, double) norm(p, p)}
     *
     * @param src MatrixOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(MatrixOld src, double p) {
        double norm;

        if(Double.isInfinite(p)) {
            if(p > 0) {
                norm = maxNorm(src);
            } else {
                norm = src.minAbs();
            }
        } else {
            norm = TensorNorms.tensorNormLp(src.entries, p);
        }

        return norm;
    }


    /**
     * Computes the maximum norm of this matrix. That is, the maximum value in the matrix.
     *
     * @param src MatrixOld to compute norm of.
     * @return The maximum norm of this matrix.
     * @see #infNorm(MatrixOld)
     */
    public static double maxNorm(MatrixOld src) {
        return matrixMaxNorm(src.entries);
    }


    /**
     * Computes the infinite norm of this matrix. that is the maximum row sum in the matrix.
     *
     * @param src MatrixOld to compute norm of.
     * @return The infinite norm of this matrix.
     * @see #maxNorm(MatrixOld)
     */
    public static double infNorm(MatrixOld src) {
        return matrixInfNorm(src.entries, src.shape);
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(MatrixOld src, double p, double q) {
        return matrixNormLpq(src.entries, src.shape, p, q);
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src MatrixOld to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CMatrixOld src, double p, double q) {
        return matrixNormLpq(src.entries, src.shape, p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src MatrixOld to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CMatrixOld src) {
        return matrixMaxNorm(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CMatrixOld, double) norm(2)}.
     *
     * @param src MatrixOld to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(CMatrixOld src) {
        return matrixNormL2(src.entries, src.shape);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src MatrixOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CMatrixOld src, double p) {
        double norm;

        if(Double.isInfinite(p)) {
            if(p > 0) {
                norm = maxNorm(src);
            } else {
                norm = src.minAbs();
            }
        } else {
            norm = matrixNormLp(src.entries, src.shape, p);
        }

        return norm;
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @param src MatrixOld to compute norm of.
     * @return The maximum/infinite norm of this tensor.
     */
    public static double infNorm(CMatrixOld src) {
        return matrixInfNorm(src.entries, src.shape);
    }


    // ------------------------------ Sparse COO Matrices ------------------------------
    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src MatrixOld to compute norm of.
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
     * @param src MatrixOld to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CooMatrix src) {
        return matrixMaxNorm(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CooMatrix, double) norm(2)}.
     *
     * @param src MatrixOld to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(CooMatrix src) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? RealSparseNorms.matrixNormL2(src) :
                norm(src.toDense());
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src MatrixOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CooMatrix src, double p) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? RealSparseNorms.matrixNormLp(src, p) :
                norm(src.toDense(), p);
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src MatrixOld to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CooCMatrix src, double p, double q) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? ComplexSparseNorms.matrixNormLpq(src, p, q) :
                norm(src.toDense(), p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src MatrixOld to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CooCMatrix src) {
        return matrixMaxNorm(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CooCMatrix, double) norm(2)}.
     *
     * @param src MatrixOld to compute the norm.
     * @return the 2-norm of this tensor.
     */
    public static double norm(CooCMatrix src) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? ComplexSparseNorms.matrixNormL2(src) :
                norm(src.toDense());
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src MatrixOld to compute the norm.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CooCMatrix src, double p) {
        // Sparse implementation is usually only faster for very sparse matrices.
        return src.sparsity()>=0.95 ? ComplexSparseNorms.matrixNormLp(src, p) :
                norm(src.toDense(), p);
    }


    // CSR Matrices

    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src MatrixOld to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CsrMatrix src, double p, double q) {
        return matrixNormLpq(src, p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src MatrixOld to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CsrMatrix src) {
        return matrixMaxNorm(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CsrMatrix, double) norm(2)}.
     *
     * @param src MatrixOld to compute the norm of.
     * @return the 2-norm of this tensor.
     */
    public double norm(CsrMatrix src) {
        return TensorNorms.tensorNormL2(src.entries); // Zeros do not contribute to this norm.
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src MatrixOld to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public double norm(CsrMatrix src, double p) {
        return TensorNorms.tensorNormLp(src.entries, p); // Zeros do not contribute to this norm.
    }


    // -------------------------------------------------- Low-level implementations --------------------------------------------------
    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    private static double matrixMaxNorm(double[] src) {
        return AggregateReal.maxAbs(src);
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum value in this matrix.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return The infinity norm of the matrix.
     */
    private static double matrixInfNorm(double[] src, Shape shape) {
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] rowSums = new double[rows];

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rowSums[i] += Math.abs(src[i*cols + j]);
            }
        }

        return AggregateReal.maxAbs(rowSums);
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
    private static double matrixNormLpq(CNumber[] src, Shape shape, double p, double q) {
        ParameterChecks.assertGreaterEq(1, p, q);

        double norm = 0;
        double colSum;
        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int j=0; j<cols; j++) {
            colSum = 0;
            for(int i=0; i<rows; i++) {
                colSum += (Math.pow(src[i*cols + j].mag(), p));
            }
            norm += Math.pow(colSum, q/p);
        }

        return Math.pow(norm, 1/q);
    }


    /**
     * Compute the L<sub>p</sub> norm of a matrix. This is equivalent to passing {@code q=1} to
     * {@link #matrixNormLpq(CNumber[], Shape, double, double)}
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @param p Parameter in L<sub>p</sub> norm.
     * @return The L<sub>p</sub> norm of the matrix.
     * @throws IllegalArgumentException If {@code p} is less than 1.
     */
    private static double matrixNormLp(CNumber[] src, Shape shape, double p) {
        ParameterChecks.assertGreaterEq(1, p);

        double norm = 0;
        double colSum;
        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int j=0; j<cols; j++) {
            colSum=0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(src[i*cols + j].mag(), p);
            }

            norm += Math.pow(colSum, 10/p);
        }

        return norm;
    }


    /**
     * Compute the L<sub>2</sub> norm of a matrix. This is equivalent to passing {@code q=1} to
     * {@link #matrixNormLpq(CNumber[], Shape, double, double)}
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return The L<sub>2</sub> norm of the matrix.
     */
    private static double matrixNormL2(CNumber[] src, Shape shape) {
        double norm = 0;
        int rows = shape.get(0);
        int cols = shape.get(1);

        double colSum;

        for(int j=0; j<cols; j++) {
            colSum = 0;
            for(int i=0; i<rows; i++) {
                colSum += Math.pow(src[i*cols + j].mag(), 2);
            }
            norm += Math.sqrt(colSum);
        }

        return norm;
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    private static double matrixMaxNorm(CNumber[] src) {
        return AggregateComplex.maxAbs(src);
    }


    /**
     * Computes the infinity/maximum norm of a matrix. That is, the maximum absolute value in this matrix.
     * @param src Entries of the matrix.
     * @return The infinity norm of the matrix.
     */
    private static double matrixInfNorm(CNumber[] src, Shape shape) {
        int rows = shape.get(0);
        int cols = shape.get(1);
        double[] rowSums = new double[rows];

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rowSums[i] += src[i*cols + j].mag();
            }
        }

        return AggregateReal.maxAbs(rowSums);
    }


    /**
     * Compute the L<sub>p,q</sub> norm of a sparse CSR matrix.
     * @param src Sparse CSR matrix to compute norm of.
     * @return The L<sub>p,q</sub> norm of the matrix.
     */
    public static double matrixNormLpq(CsrMatrix src, double p, double q) {
        CsrMatrix tSrc = src.transpose();
        double norm = 0;
        double pOverQ = p/q;

        for(int i=0; i<tSrc.numRows; i++) {
            int start = tSrc.rowPointers[i];
            int stop = tSrc.rowPointers[i+1];
            double colNorm = 0;

            for(int j=start; j<stop; j++) {
                colNorm += Math.pow(Math.abs(tSrc.entries[j]), p);
            }

            norm += Math.pow(colNorm, pOverQ);
        }

        return Math.pow(norm, 10/q);
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
    public static double matrixNormLpq(double[] src, Shape shape, double p, double q) {
        ParameterChecks.assertGreaterEq(1, p, q);

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
}
