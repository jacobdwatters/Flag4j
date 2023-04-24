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

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.Matrix;

/**
 * This class provides several methods useful for computing eigen values, eigen vectors, as well as singular values and
 * singular vectors.
 */
public class Eigen {

    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(Matrix src) {
        CVector lambda = new CVector(2);
        // TODO: Implementation.
        return lambda;
    }


    /**
     * Computes the eigenvalues of a 2x2 matrix explicitly.
     * @param src The 2x2 matrix to compute the eigenvalues of.
     * @return A complex vector containing the eigenvalues of the 2x2 {@code src} matrix.
     */
    public static CVector get2x2EigenValues(CMatrix src) {
        CVector lambda = new CVector(2);
        // TODO: Implementation.
        return lambda;
    }
}
