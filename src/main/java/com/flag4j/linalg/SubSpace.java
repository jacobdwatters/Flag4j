/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

package com.flag4j.linalg;


import com.flag4j.Matrix;
import com.flag4j.linalg.decompositions.RealSVD;
import com.flag4j.util.ErrorMessages;

/**
 * This class contains several methods for computing the subspace of a matrix.
 */
public class SubSpace {
    // TODO: Implementation of rowSpace, colSpace, nullSpace and leftNullSpace for each matrix type.

    private SubSpace() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes an orthonormal basis of the column space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the column space.
     * @return A matrix containing as its columns, an orthonormal basis for the column space of the {@code src} matrix.
     */
    public static Matrix getColumnSpace(Matrix src) {
        RealSVD svd = new RealSVD().decompose(src);
        int rank = rankFromSVD(svd.getS());
        return svd.getU().getSlice(0, src.numRows, 0, rank);
    }


    /**
     * Computes an orthonormal basis of the row space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the row space.
     * @return A matrix containing as its columns, an orthonormal basis for the row space of the {@code src} matrix.
     */
    public static Matrix getRowSpace(Matrix src) {
        RealSVD svd = new RealSVD().decompose(src);
        int rank = rankFromSVD(svd.getS());
        return svd.getV().getSlice(0, src.numRows, 0, rank);
    }


    /**
     * Computes an orthonormal basis of the null space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the null space.
     * @return A matrix containing as its columns, an orthonormal basis for the null space of the {@code src} matrix.
     */
    public static Matrix getNullSpace(Matrix src) {
        RealSVD svd = new RealSVD().decompose(src);
        int rank = rankFromSVD(svd.getS());
        return svd.getV().getSlice(0, src.numCols, rank+1, src.numCols);
    }


    /**
     * Computes an orthonormal basis of the left null space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the left null space.
     * @return A matrix containing as its columns, an orthonormal basis for the left null space of the {@code src} matrix.
     */
    public static Matrix getLeftNullSpace(Matrix src) {
        RealSVD svd = new RealSVD().decompose(src);
        int rank = rankFromSVD(svd.getS());
        return svd.getU().getSlice(0, src.numRows, rank+1, src.numRows);
    }


    /**
     * Computes the rank of a matrix given the matrix {@code S} from the SVD {@code M=USV<sup>T</sup>}.
     * @param S The matrix {@code S} from the SVD {@code M=USV<sup>T</sup>}.
     * @return the rank of the matrix {@code M} from the SVD {@code M=USV<sup>T</sup>}.
     */
    private static int rankFromSVD(Matrix S) {
        int stopIdx = Math.min(S.numRows, S.numCols);

        double tol = 1.0E-8; // Tolerance for determining if a singular value should be considered zero.
        int rank = 0;

        for(int i=0; i<stopIdx; i++) {
            if(S.get(i, i)>tol) {
                rank++;
            }
        }

        return rank;
    }
}
