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

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.util.ErrorMessages;


/**
 * Utility class for computing the condition number of a matrix.
 */
public class Condition {

    private Condition() {
        // Hide default constructor for utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the condition number of this matrix using the 2-norm.
     * Specifically, the condition number is computed as the norm of this matrix multiplied by the norm
     * of the inverse of this matrix.
     *
     * @param src Matrix to compute the condition number of.
     * @return The condition number of this matrix (Assuming 2-norm). This value may be
     * {@link Double#POSITIVE_INFINITY infinite}.
     */
    public static double cond(Matrix src) {
        return cond(src, 2);
    }


    /**
     * Computes the condition number of this matrix using a specified norm. The condition number of a matrix is defined
     * as the norm of a matrix multiplied by the norm of the inverse of the matrix.
     * @param src Matrix to compute the condition number of.
     * @param p Specifies the order of the norm to be used when computing the condition number.
     *          Common {@code p} values include:<br>
     *          - {@code p} = {@link Double#POSITIVE_INFINITY}, {@link MatrixNorms#infNorm(Matrix)}.<br>
     *          - {@code p} = 2, The standard matrix 2-norm (the largest singular value).<br>
     *          - {@code p} = -2, The Smallest singular value.<br>
     *          - {@code p} = 1, Maximum absolute row sum.<br>
     * @return The condition number of this matrix using the specified norm. This value may be
     * {@link Double#POSITIVE_INFINITY infinite}.
     */
    public static double cond(Matrix src, double p) {
        double cond;

        if(p==2 || p==-2) {
            // Compute the singular value decomposition of the matrix.
            Vector s = new RealSVD(false).decompose(src).getS().getDiag();
            cond = p==2 ? s.max()/s.min() : s.min()/s.max();
        } else {
            cond = MatrixNorms.norm(src, p)*MatrixNorms.norm(Invert.inv(src), p);
        }

        return cond;
    }


    /**
     * Computes the condition number of this matrix using the 2-norm.
     * Specifically, the condition number is computed as the norm of this matrix multiplied by the norm
     * of the inverse of this matrix.
     *
     * @param src Matrix to compute the condition number of.
     * @return The condition number of this matrix (Assuming 2-norm). This value may be
     * {@link Double#POSITIVE_INFINITY infinite}.
     */
    public static double cond(CMatrix src) {
        return cond(src,2);
    }


    /**
     * Computes the condition number of this matrix using a specified norm. The condition number of a matrix is defined
     * as the norm of a matrix multiplied by the norm of the inverse of the matrix.
     * @param src Matrix to compute the condition number of.
     * @param p Specifies the order of the norm to be used when computing the condition number.
     *          Common {@code p} values include:<br>
     *          - {@code p} = {@link Double#POSITIVE_INFINITY}, {@link MatrixNorms#infNorm(CMatrix)}.<br>
     *          - {@code p} = 2, The standard matrix 2-norm (the largest singular value).<br>
     *          - {@code p} = -2, The Smallest singular value.<br>
     *          - {@code p} = 1, Maximum absolute row sum.<br>
     * @return The condition number of this matrix using the specified norm. This value may be
     * {@link Double#POSITIVE_INFINITY infinite}.
     */
    public static double cond(CMatrix src, double p) {
        double cond;

        if(p==2 || p==-2) {
            // Compute the singular value decomposition of the matrix.
            Vector s = new ComplexSVD(false).decompose(src).getS().getDiag();
            cond = p==2 ? s.max()/s.min() : s.min()/s.max();
        } else {
            cond = MatrixNorms.norm(src, p) * MatrixNorms.norm(Invert.inv(src), p);
        }

        return cond;
    }
}
