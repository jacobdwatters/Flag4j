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


import com.flag4j.Matrix;
import com.flag4j.core.MatrixMixin;

/**
 * This abstract class specifies methods for computing the singular value decomposition (SVD) of a matrix.
 * That is, decompose a rectangular matrix {@code M} as {@code M=USV<sup>H</sup>} where {@code U} and {@code V} are
 * unitary matrices whose columns are the left and right singular vectors of {@code M} and {@code S} is a rectangular
 * diagonal matrix containing the singular values of {@code M}.
 * @param <T> The type of the matrix to compute the singular value decomposition of.
 */
public abstract class SingularValueDecomposition<T extends MatrixMixin<T, ?, ?, ?, ?, ?, ?>> implements Decomposition<T> {

    /**
     * Flag which indicates if the singular vectors should be computed in addition to the singular values.
     */
    protected boolean computeUV;

    /**
     * The unitary matrix {@code U} corresponding to {@code M=USV<sup>H</sup>} in the SVD.
     */
    protected T U;
    /**
     * The rectangular diagonal {@code S} corresponding to {@code M=USV<sup>H</sup>} in the SVD.
     */
    protected Matrix S;
    /**
     * The unitary matrix {@code V} corresponding to {@code M=USV<sup>H</sup>} in the SVD.
     */
    protected T V;


    /**
     * Creates a decomposer to compute the Schur decomposition.
     * @param computeUV A flag which indicates if the unitary matrices {@code Q} and {@code V} should be computed
     *                  (i.e. the singular vectors).<br>
     *                 - If true, the {@code Q} and {@code V} matrices will be computed.
     *                 - If false, the {@code Q} and {@code V} matrices  will <b>not</b> be computed. If it is not needed, this may
     *                 provide a performance improvement.
     */
    protected SingularValueDecomposition(boolean computeUV) {
        this.computeUV = computeUV;
    }


    /**
     * Gets the unitary matrix {@code U} corresponding to {@code M=USV<sup>H</sup>} in the SVD.
     * @return {@code U} corresponding to {@code M=USV<sup>H</sup>} in the SVD.
     */
    public T getU() {
        return U;
    }


    /**
     * Gets the diagonal matrix {@code S} corresponding to {@code M=USV<sup>H</sup>} in the SVD.
     * @return {@code S} corresponding to {@code M=USV<sup>H</sup>} in the SVD.
     */
    public Matrix getS() {
        return S;
    }


    /**
     * Gets the unitary matrix {@code V} corresponding to {@code M=USV<sup>H</sup>} in the SVD.
     * @return {@code V} corresponding to {@code M=USV<sup>H</sup>} in the SVD. Note that the hermation transpose has
     * <b>not</b> been computed.
     */
    public T getV() {
        return V;
    }
}
