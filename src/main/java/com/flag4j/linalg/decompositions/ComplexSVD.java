/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.linalg.decompositions;


import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.linalg.Eigen;

/**
 * Instances of this class can be used to compute the singular value decomposition (SVD) of a real dense matrix.
 * That is, decompose a rectangular matrix {@code M} as {@code M=USV<sup>H</sup>} where {@code U} and {@code V} are
 * unitary matrices whose columns are the left and right singular vectors of {@code M} and {@code S} is a rectangular
 * diagonal matrix containing the singular values of {@code M}.
 */
public class ComplexSVD extends SingularValueDecomposition<CMatrix> {


    /**
     * Creates a decomposer to compute the Schur decomposition.
     *
     * @param computeUV A flag which indicates if the unitary matrices {@code Q} and {@code V} should be computed
     *                  (i.e. the singular vectors).<br>
     *                  - If true, the {@code Q} and {@code V} matrices will be computed.<br>
     *                  - If false, the {@code Q} and {@code V} matrices  will <b>not</b> be computed. If it is not needed, this may
     *                  provide a performance improvement.
     */
    protected ComplexSVD(boolean computeUV) {
        super(computeUV);
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public Decomposition<CMatrix> decompose(CMatrix src) {
        Vector S1;

        if(computeUV) {
            // Compute right singular vectors.
            CMatrix[] pairs = Eigen.getEigenPairs(src.H().mult(src));
            V = pairs[1];

            // Compute left singular vectors.
            U = src.mult(V);

            // Compute singular values.
            S1 = pairs[0].toVector().toReal().abs().sqrt();

            // Copy singular values to diagonal of S and scale left singular vectors properly.
        } else {
            S1 = Eigen.getEigenValues(src.H().mult(src)).toReal();
        }

        S = new Matrix(src.shape);
        for(int i=0; i<S1.size; i++) {
            S.set(S1.entries[i], i, i);

            if(computeUV && S1.entries[i] != 0) {
                // Properly scale the left singular vectors so that Mv = su.
                divCols(i, S1.entries[i]);
            }
        }

        return this;
    }


    /**
     * Divides a specified column of a matrix by a scalar value.
     * @param colIdx Index of column to divide.
     * @param scalValue Value to divide column by.
     */
    private void divCols(int colIdx, double scalValue) {
        int idx = colIdx;
        for(int i=0; i<U.numRows; i++) {
            U.entries[idx] = U.entries[idx].div(scalValue);
            idx+=U.numCols;
        }
    }
}
