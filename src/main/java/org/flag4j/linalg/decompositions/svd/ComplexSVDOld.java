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

package org.flag4j.linalg.decompositions.svd;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.linalg.DirectSumOld;
import org.flag4j.linalg.EigenOld;


/**
 * Instances of this class can be used to compute the singular value decomposition (SVDOld) of a real dense matrix.
 * That is, decompose a rectangular matrix {@code M} as {@code M=USV<sup>H</sup>} where {@code U} and {@code V} are
 * unitary matrices whose columns are the left and right singular vectors of {@code M} and {@code S} is a rectangular
 * diagonal matrix containing the singular values of {@code M}.
 */
@Deprecated
public class ComplexSVDOld extends SVDOld<CMatrixOld> {


    /**
     * Creates a decomposer to compute the singular value decomposition of a real matrix. The left and right singular
     * vectors will be computed.
     */
    public ComplexSVDOld() {
        super(true, false);
    }


    /**
     * Creates a decomposer to compute the SchurOld decomposition.
     *
     * @param computeUV A flag which indicates if the unitary matrices {@code Q} and {@code V} should be computed
     *                  (i.e. the singular vectors). By default, this is true.<br>
     *                  - If true, the {@code Q} and {@code V} matrices will be computed.<br>
     *                  - If false, the {@code Q} and {@code V} matrices  will <b>not</b> be computed. If it is not needed, this may
     *                  provide a performance improvement.
     */
    public ComplexSVDOld(boolean computeUV) {
        super(computeUV, false);
    }


    /**
     * Creates a decomposer to compute the singular value decomposition of a real matrix.
     * @param computeUV A flag which indicates if the orthogonal matrices {@code Q} and {@code V} should be computed
     *                  (i.e. the singular vectors). By default, this is true.<br>
     *                 - If true, the {@code Q} and {@code V} matrices will be computed.<br>
     *                 - If false, the {@code Q} and {@code V} matrices  will <b>not</b> be computed. If they are not
     *                 needed, this may provide a performance improvement.
     * @param reduced Flag which indicates if the reduced (or full) SVDOld should be computed. This is false by default.<br>
     *                 - If true, reduced SVDOld is computed.
     *                 - If false, the full SVDOld is computed.
     */
    public ComplexSVDOld(boolean computeUV, boolean reduced) {
        super(computeUV, reduced);
    }


    /**
     * Computes the inverse direct sum of a matrix and its hermitian transpose.
     *
     * @param src MatrixOld to inverse direct add with its hermitian transpose.
     *
     * @return The inverse direct sum of the {@code src} matrix with its hermitian transpose.
     */
    @Override
    protected CMatrixOld invDirectSum(CMatrixOld src) {
        return DirectSumOld.directSum(src, src.H());
    }


    /**
     * Gets the eigen values and vectors of symmetric the block matrix which corresponds
     * to the singular values and vectors of the matrix being decomposed.
     *
     * @param B       Symmetric block matrix to compute the eigenvalues of.
     * @param eigVals Storage for eigenvalues.
     * @return The eigenvalues and eigenvectors of the symmetric block matrix which corresponds
     * to the singular values and vectors of the matrix being decomposed.
     */
    @Override
    protected CMatrixOld makeEigenPairs(CMatrixOld B, double[] eigVals) {
        CMatrixOld[] pairs = EigenOld.getEigenPairs(B);

        double[] vals = pairs[0].toReal().entries;
        System.arraycopy(vals, 0, eigVals, 0, eigVals.length);

        return pairs[1];
    }


    /**
     * Gets the eigen values of the symmetric block matrix which corresponds
     * to the singular values of the matrix being decomposed.
     *
     * @param B       Symmetric block matrix to compute the eigenvalues of.
     * @param eigVals Storage for eigenvalues.
     */
    @Override
    protected void makeEigenVals(CMatrixOld B, double[] eigVals) {
        double[] vals = EigenOld.getEigenValues(B).toReal().entries;
        System.arraycopy(vals, 0, eigVals, 0, eigVals.length);
    }


    /**
     * Initializes the unitary {@code U} and {@code V} matrices for the SVDOld.
     *
     * @param src  Shape of the source matrix being decomposed.
     * @param cols The number of columns for {@code U} and {@code V}.
     */
    @Override
    protected void initUV(Shape src, int cols) {
        U = new CMatrixOld(src.get(0), cols);
        V = new CMatrixOld(src.get(1), cols);
    }


    /**
     * Extracts the singular vectors, normalizes them and sets the columns of {@code U}
     * and {@code V} to be the left/right singular vectors.
     *
     * @param singularVecs Computed left and right singular vectors.
     * @param j            Index of the column of {@code U} and {@code V} to set.
     */
    @Override
    protected void extractNormalizedCols(CMatrixOld singularVecs, int j) {
        // Extract left and right singular vectors and normalize.
        V.setCol(singularVecs.getCol(2*j, 0, V.numRows()).normalize(), j);
        U.setCol(singularVecs.getCol(2*j, V.numRows(), singularVecs.numRows()).normalize(), j);
    }
}
