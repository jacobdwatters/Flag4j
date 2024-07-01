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


import org.flag4j.dense.CMatrix;
import org.flag4j.dense.Matrix;
import org.flag4j.operations.dense.complex.ComplexDenseOperations;
import org.flag4j.operations.dense.real.RealDenseOperations;
import org.flag4j.operations.sparse.coo.complex.ComplexSparseNorms;
import org.flag4j.operations.sparse.coo.real.RealSparseNorms;
import org.flag4j.operations.sparse.csr.real.RealCsrOperations;
import org.flag4j.sparse.CooCMatrix;
import org.flag4j.sparse.CooMatrix;
import org.flag4j.sparse.CsrMatrix;
import org.flag4j.util.ErrorMessages;

/**
 * Utility class for computing norms of matrices.
 */
public class MatrixNorms {

    private MatrixNorms() {
        // Hide default constructor for utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(Matrix, double) norm(2)}.
     * This will be equal to the largest singular value of the matrix.
     *
     * @param src Matrix to compute norm of.
     *
     * @return the 2-norm of this tensor.
     */
    public static double norm(Matrix src) {
        return RealDenseOperations.tensorNormL2(src.entries);
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
        double norm;

        if(Double.isInfinite(p)) {
            if(p > 0) {
                norm = maxNorm(src);
            } else {
                norm = src.minAbs();
            }
        } else {
            norm = RealDenseOperations.tensorNormLp(src.entries, p);
        }

        return norm;
    }


    /**
     * Computes the maximum norm of this matrix. That is, the maximum value in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The maximum norm of this matrix.
     * @see #infNorm(Matrix)
     */
    public static double maxNorm(Matrix src) {
        return RealDenseOperations.matrixMaxNorm(src.entries);
    }


    /**
     * Computes the infinite norm of this matrix. that is the maximum row sum in the matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The infinite norm of this matrix.
     * @see #maxNorm(Matrix)
     */
    public static double infNorm(Matrix src) {
        return RealDenseOperations.matrixInfNorm(src.entries, src.shape);
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(Matrix src, double p, double q) {
        return RealDenseOperations.matrixNormLpq(src.entries, src.shape, p, q);
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param src Matrix to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CMatrix src, double p, double q) {
        return ComplexDenseOperations.matrixNormLpq(src.entries, src.shape, p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CMatrix src) {
        return ComplexDenseOperations.matrixMaxNorm(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CMatrix, double) norm(2)}.
     *
     * @param src Matrix to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(CMatrix src) {
        return ComplexDenseOperations.matrixNormL2(src.entries, src.shape);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src Matrix to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(CMatrix src, double p) {
        double norm;

        if(Double.isInfinite(p)) {
            if(p > 0) {
                norm = maxNorm(src);
            } else {
                norm = src.minAbs();
            }
        } else {
            norm = ComplexDenseOperations.matrixNormLp(src.entries, src.shape, p);
        }

        return norm;
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @param src Matrix to compute norm of.
     * @return The maximum/infinite norm of this tensor.
     */
    public static double infNorm(CMatrix src) {
        return ComplexDenseOperations.matrixInfNorm(src.entries, src.shape);
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
        return RealDenseOperations.matrixMaxNorm(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CooMatrix, double) norm(2)}.
     *
     * @param src Matrix to compute norm of.
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
     * @param src Matrix to compute norm of.
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
     * @param src Matrix to compute norm of.
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
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CooCMatrix src) {
        return ComplexDenseOperations.matrixMaxNorm(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CooCMatrix, double) norm(2)}.
     *
     * @param src Matrix to compute the norm.
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
     * @param src Matrix to compute the norm.
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
     * @param src Matrix to compute norm of.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    public static double norm(CsrMatrix src, double p, double q) {
        return RealCsrOperations.matrixNormLpq(src, p, q);
    }


    /**
     * Computes the max norm of a matrix.
     *
     * @param src Matrix to compute norm of.
     * @return The max norm of this matrix.
     */
    public static double maxNorm(CsrMatrix src) {
        return RealDenseOperations.matrixMaxNorm(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CsrMatrix, double) norm(2)}.
     *
     * @param src Matrix to compute the norm of.
     * @return the 2-norm of this tensor.
     */
    public double norm(CsrMatrix src) {
        return RealDenseOperations.tensorNormL2(src.entries); // Zeros do not contribute to this norm.
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src Matrix to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public double norm(CsrMatrix src, double p) {
        return RealDenseOperations.tensorNormLp(src.entries, p); // Zeros do not contribute to this norm.
    }
}
