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
import org.flag4j.linalg.decompositions.chol.CholeskyOld;
import org.flag4j.linalg.decompositions.chol.ComplexCholeskyOld;
import org.flag4j.linalg.decompositions.chol.RealCholeskyOld;
import org.flag4j.linalg.decompositions.hess.RealHessOld;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.decompositions.lu.LUOld;
import org.flag4j.linalg.decompositions.lu.RealLUOLd;
import org.flag4j.linalg.decompositions.qr.ComplexQROld;
import org.flag4j.linalg.decompositions.qr.RealQROld;
import org.flag4j.linalg.decompositions.schur.RealSchurOld;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.util.ErrorMessages;

/**
 * A factory class for creating decomposers to perform various matrix decompositions.
 */
public class DecompositionFactory {

    private DecompositionFactory() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Constructs a decomposer to compute the LUOld decomposition of a real dense matrix.
     * @return A decomposer to compute the LUOld decomposition of a real dense matrix.
     */
    public static LUOld<MatrixOld> createRealLU() {
        return new RealLUOLd();
    }


    /**
     * Constructs a decomposer to compute the LUOld decomposition of a complex dense matrix.
     * @return A decomposer to compute the LUOld decomposition of a complex dense matrix.
     */
    public static LUOld<CMatrixOld> createComplexLU() {
        return new ComplexLU();
    }


    /**
     * Constructs a decomposer to compute the CholeskyOld decomposition of a real dense matrix.
     * @return A decomposer to compute the CholeskyOld decomposition of a real dense matrix.
     */
    public static CholeskyOld<MatrixOld> createRealChol() {
        return new RealCholeskyOld();
    }


    /**
     * Constructs a decomposer to compute the CholeskyOld decomposition of a complex dense matrix.
     * @return A decomposer to compute the CholeskyOld decomposition of a complex dense matrix.
     */
    public static CholeskyOld<CMatrixOld> createComplexChol() {
        return new ComplexCholeskyOld();
    }


    /**
     * Constructs a decomposer to compute the QR decomposition of a real dense matrix.
     * @return A decomposer to compute the QR decomposition of a real dense matrix.
     */
    public static RealQROld createRealQR() {
        return new RealQROld();
    }


    /**
     * Constructs a decomposer to compute the QR decomposition of a complex dense matrix.
     * @return A decomposer to compute the QR decomposition of a complex dense matrix.
     */
    public static ComplexQROld createComplexQR() {
        return new ComplexQROld();
    }


    /**
     * Constructs a decomposer to compute the Hessenburg decomposition of a real dense matrix.
     * @return A decomposer to compute the Hessenburg decomposition of a real dense matrix.
     */
    public static RealHessOld createRealHess() {
        return new RealHessOld();
    }


    /**
     * Constructs a decomposer to compute the Hessenburg decomposition of a complex dense matrix.
     * @return A decomposer to compute the Hessenburg decomposition of a complex dense matrix.
     */
    public static ComplexQROld createComplexHess() {
        return new ComplexQROld();
    }


    /**
     * Constructs a decomposer to compute the SchurOld decomposition of a real dense matrix.
     * @return A decomposer to compute the SchurOld decomposition of a real dense matrix.
     */
    public static RealSchurOld createRealSchur() {
        return new RealSchurOld();
    }


    /**
     * Constructs a decomposer to compute the SchurOld decomposition of a complex dense matrix.
     * @return A decomposer to compute the SchurOld decomposition of a complex dense matrix.
     */
    public static ComplexQROld createComplexSchur() {
        return new ComplexQROld();
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
