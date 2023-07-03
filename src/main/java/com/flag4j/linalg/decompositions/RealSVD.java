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
import com.flag4j.linalg.Eigen;

/**
 * Instances of this class can be used to compute the singular value decomposition (SVD) of a real dense matrix.
 * That is, decompose a rectangular matrix {@code M} as {@code M=USV<sup>T</sup>} where {@code U} and {@code V} are
 * orthogonal matrices whose columns are the left and right singular vectors of {@code M} and {@code S} is a rectangular
 * diagonal matrix containing the singular values of {@code M}.
 */
public class RealSVD extends SingularValueDecomposition<Matrix> {

    /**
     * Creates a decomposer to compute the singular value decomposition of a real matrix. The left and right singular
     * vectors will be computed.
     */
    public RealSVD() {
        super(true);
    }


    /**
     * Creates a decomposer to compute the singular value decomposition of a real matrix.
     * @param computeUV A flag which indicates if the orthogonal matrices {@code Q} and {@code V} should be computed
     *                  (i.e. the singular vectors).<br>
     *                 - If true, the {@code Q} and {@code V} matrices will be computed.<br>
     *                 - If false, the {@code Q} and {@code V} matrices  will <b>not</b> be computed. If they are not
     *                 needed, this may provide a performance improvement.
     */
    public RealSVD(boolean computeUV) {
        super(computeUV);
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public RealSVD decompose(Matrix src) {
        Matrix B = src.invDirectSum(src.T());
        double[] eigVals;

        int stopIdx = Math.min(src.numRows, src.numCols);

        if(computeUV) {
            CMatrix[] pairs = Eigen.getEigenPairs(B);

            eigVals = pairs[0].toReal().entries;
            Matrix eigVecs = pairs[1].toReal();

            U = new Matrix(src.numRows);
            V = new Matrix(src.numCols);

            for(int j=0; j<stopIdx; j++) {
                // Extract left and right singular vectors and normalize.
                V.setCol(eigVecs.getCol(2*j, 0, V.numRows).normalize(), j);
                U.setCol(eigVecs.getCol(2*j, V.numRows, eigVecs.numRows).normalize(), j);
            }

        } else {
            eigVals = Eigen.getEigenValues(B).toReal().entries;
        }

        S = new Matrix(src.shape);
        for(int i=0; i<stopIdx; i++) {
            S.set(eigVals[2*i], i, i);
        }

        return this;
    }


    /**
     * Divides a specified column of the {@code U} matrix in the SVD by a scalar value.
     * @param colIdx Index of column to divide.
     * @param scalValue Value to divide column by.
     */
    private void divCols(int colIdx, double scalValue) {
        int idx = colIdx;
        for(int i=0; i<U.numRows; i++) {
            U.entries[idx] /= scalValue;
            idx+=U.numCols;
        }
    }
}
