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
        super(true, false);
    }


    /**
     * Creates a decomposer to compute the singular value decomposition of a real matrix.
     * @param computeUV A flag which indicates if the orthogonal matrices {@code Q} and {@code V} should be computed
     *                  (i.e. the singular vectors). By default, this is true.<br>
     *                 - If true, the {@code Q} and {@code V} matrices will be computed.<br>
     *                 - If false, the {@code Q} and {@code V} matrices  will <b>not</b> be computed. If they are not
     *                 needed, this may provide a performance improvement.
     */
    public RealSVD(boolean computeUV) {
        super(computeUV, false);
    }


    /**
     * Creates a decomposer to compute the singular value decomposition of a real matrix.
     * @param computeUV A flag which indicates if the orthogonal matrices {@code Q} and {@code V} should be computed
     *                  (i.e. the singular vectors). By default, this is true.<br>
     *                 - If true, the {@code Q} and {@code V} matrices will be computed.<br>
     *                 - If false, the {@code Q} and {@code V} matrices  will <b>not</b> be computed. If they are not
     *                 needed, this may provide a performance improvement.
     * @param reduced Flag which indicates if the reduced (or full) SVD should be computed. This is false by default.<br>
     *                 - If true, reduced SVD is computed.
     *                 - If false, the full SVD is computed.
     */
    public RealSVD(boolean computeUV, boolean reduced) {
        super(computeUV, reduced);
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public RealSVD decompose(Matrix src) {
        double[] singularVals;
        int stopIdx;
        Matrix singularVecs = null;

        Matrix B = src.invDirectSum(src.H()); // Convert the problem to an eigenvalue problem.

        if(computeUV) {
            CMatrix[] pairs = Eigen.getEigenPairs(B);

            singularVals = pairs[0].toReal().entries;
            singularVecs = pairs[1].toReal();
        } else {
            singularVals = Eigen.getEigenValues(B).toReal().entries;
        }

        computeRank(src.numRows, src.numCols, singularVals);
        stopIdx = reduced ? rank : Math.min(src.numRows, src.numCols);

        S = new Matrix(stopIdx);

        if(computeUV) {
            U = new Matrix(src.numRows, stopIdx);
            V = new Matrix(src.numCols, stopIdx);
        }

        for(int j=0; j<stopIdx; j++) {
            S.set(singularVals[2*j], j, j);

            if(computeUV && singularVecs != null) {
                // Extract left and right singular vectors and normalize.
                V.setCol(singularVecs.getCol(2*j, 0, V.numRows).normalize(), j);
                U.setCol(singularVecs.getCol(2*j, V.numRows, singularVecs.numRows).normalize(), j);
            }
        }

        return this;
    }
}
