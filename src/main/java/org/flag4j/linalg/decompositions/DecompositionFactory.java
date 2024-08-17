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

package org.flag4j.linalg.decompositions;


import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.linalg.decompositions.chol.Cholesky;
import org.flag4j.linalg.decompositions.chol.ComplexCholesky;
import org.flag4j.linalg.decompositions.chol.RealCholesky;
import org.flag4j.linalg.decompositions.hess.RealHess;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.linalg.decompositions.lu.RealLU;
import org.flag4j.linalg.decompositions.qr.ComplexQR;
import org.flag4j.linalg.decompositions.qr.RealQR;
import org.flag4j.linalg.decompositions.schur.RealSchur;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.util.ErrorMessages;

/**
 * A factory class for creating decomposers to perform various matrix decompositions.
 */
public class DecompositionFactory {

    private DecompositionFactory() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Constructs a decomposer to compute the LU decomposition of a real dense matrix.
     * @return A decomposer to compute the LU decomposition of a real dense matrix.
     */
    public static LU<MatrixOld> createRealLU() {
        return new RealLU();
    }


    /**
     * Constructs a decomposer to compute the LU decomposition of a complex dense matrix.
     * @return A decomposer to compute the LU decomposition of a complex dense matrix.
     */
    public static LU<CMatrixOld> createComplexLU() {
        return new ComplexLU();
    }


    /**
     * Constructs a decomposer to compute the Cholesky decomposition of a real dense matrix.
     * @return A decomposer to compute the Cholesky decomposition of a real dense matrix.
     */
    public static Cholesky<MatrixOld> createRealChol() {
        return new RealCholesky();
    }


    /**
     * Constructs a decomposer to compute the Cholesky decomposition of a complex dense matrix.
     * @return A decomposer to compute the Cholesky decomposition of a complex dense matrix.
     */
    public static Cholesky<CMatrixOld> createComplexChol() {
        return new ComplexCholesky();
    }


    /**
     * Constructs a decomposer to compute the QR decomposition of a real dense matrix.
     * @return A decomposer to compute the QR decomposition of a real dense matrix.
     */
    public static RealQR createRealQR() {
        return new RealQR();
    }


    /**
     * Constructs a decomposer to compute the QR decomposition of a complex dense matrix.
     * @return A decomposer to compute the QR decomposition of a complex dense matrix.
     */
    public static ComplexQR createComplexQR() {
        return new ComplexQR();
    }


    /**
     * Constructs a decomposer to compute the Hessenburg decomposition of a real dense matrix.
     * @return A decomposer to compute the Hessenburg decomposition of a real dense matrix.
     */
    public static RealHess createRealHess() {
        return new RealHess();
    }


    /**
     * Constructs a decomposer to compute the Hessenburg decomposition of a complex dense matrix.
     * @return A decomposer to compute the Hessenburg decomposition of a complex dense matrix.
     */
    public static ComplexQR createComplexHess() {
        return new ComplexQR();
    }


    /**
     * Constructs a decomposer to compute the Schur decomposition of a real dense matrix.
     * @return A decomposer to compute the Schur decomposition of a real dense matrix.
     */
    public static RealSchur createRealSchur() {
        return new RealSchur();
    }


    /**
     * Constructs a decomposer to compute the Schur decomposition of a complex dense matrix.
     * @return A decomposer to compute the Schur decomposition of a complex dense matrix.
     */
    public static ComplexQR createComplexSchur() {
        return new ComplexQR();
    }


    /**
     * Constructs a decomposer to compute the singular value decomposition of a real dense matrix.
     * @return A decomposer to compute the singular value decomposition of a real dense matrix.
     */
    public static RealSVD createRealSVD() {
        return new RealSVD();
    }


    /**
     * Constructs a decomposer to compute the singular value decomposition of a complex dense matrix.
     * @return A decomposer to compute the singular value decomposition of a complex dense matrix.
     */
    public static ComplexSVD createComplexSVD() {
        return new ComplexSVD();
    }
}
